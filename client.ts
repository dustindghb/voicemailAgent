import { Ollama } from 'ollama';
import { ChromaClient } from 'chromadb';
import dotenv from 'dotenv';

// Load environment variables from .env file
dotenv.config();

// Type definitions to fix TypeScript errors
interface VoicemailTranscript {
  id: string;
  from: string;
  transcript: string;
}

interface CallerInfo {
  name: string | null;
  company: string | null;
  phone: string[] | null;
}

// Define ChromaDB specific types
type ChromaMetadata = Record<string, string | number | boolean>;
type ChromaEmbedding = number[];

interface OllamaEmbeddingResponse {
  embedding: number[];
}

// Get configuration from environment variables
const OLLAMA_HOST = process.env.OLLAMA_HOST || 'http://localhost:11434';
const CHROMA_HOST = process.env.CHROMA_HOST || 'http://localhost:8000';

// Create Ollama client with host from environment variables
const ollama = new Ollama({ host: OLLAMA_HOST });

// Create ChromaDB client with host from environment variables
const chromaClient = new ChromaClient({ path: CHROMA_HOST });

// Sample voicemail transcripts (replace with your actual data source)
const voicemailTranscripts: VoicemailTranscript[] = [
    {
      id: "vm-001",
      from: "Unknown",
      transcript: "Hi John, this is Mark from Acme Corp. I'm following up on our discussion last week about the partnership agreement. We've reviewed the terms on our end and our legal team has a few concerns about section 3.2 regarding IP ownership. Could you give me a call back when you have a chance? My direct line is 555-123-4567. Thanks, I look forward to moving this forward."
    },
    {
      id: "vm-002",
      from: "Unknown",
      transcript: "Hello, this is Dr. Peterson's office calling to reschedule your quarterly business medical checkup. We had you down for next Tuesday at 10 AM, but the doctor will be at a conference that day. We'd like to move you to Thursday same week at 9 AM if that works for you. Please call us back at 555-867-5309 to confirm or find another time. Thank you."
    },
    {
      id: "vm-003",
      from: "Unknown",
      transcript: "Hey there, it's Sarah from TechBridge Solutions. I wanted to discuss the upcoming software deployment scheduled for next weekend. Our team has identified a potential issue with the database migration that might cause some downtime. Could you call me back at 555-789-1234 to discuss mitigation strategies? This is somewhat urgent as we need to finalize the plan by Friday."
    },
    {
      id: "vm-004",
      from: "Unknown",
      transcript: "This is Michael Johnson from Legal Advisors Inc. I've prepared the contract revisions you requested last month. All the changes to the liability clauses have been implemented as discussed. When you have time, please review the document I emailed and let me know if any further modifications are needed. You can reach me at 555-321-9876 or through my assistant at extension 4321."
    },
    {
      id: "vm-005",
      from: "Unknown", 
      transcript: "Hi, this is Emma from Marketing Innovations. Just following up on the campaign proposal we submitted last week. Our creative team has some additional mock-ups they'd like to share before the deadline. I think these new designs better align with your brand guidelines. I'm available for a quick call tomorrow afternoon at your convenience. My number is 555-444-7890."
    }
    // Add your actual voicemail transcripts here
  ];

// Function to get embeddings using your remote Nomic model through Ollama
async function getEmbedding(text: string): Promise<ChromaEmbedding | null> {
  try {
    const response = await ollama.embeddings({
      model: 'nomic-embed-text',
      prompt: text
    }) as OllamaEmbeddingResponse;
    
    return response.embedding;
  } catch (error: any) {
    console.error('Error getting embedding:', error.message);
    return null;
  }
}

// Function to extract caller information (reused from your example)
async function extractCallerInfo(transcript: string): Promise<CallerInfo> {
  const callerPatterns = [
    { regex: /this is ([A-Z][a-z]+ ?([A-Z][a-z]+)?) from ([A-Za-z0-9\s&]+)/i, groups: { name: 1, company: 3 } },
    { regex: /(?:it'?s|it is) ([A-Z][a-z]+ ?([A-Z][a-z]+)?) from ([A-Za-z0-9\s&]+)/i, groups: { name: 1, company: 3 } },
    { regex: /(?:this is|it'?s|it is) ([A-Z][a-z]+ ?([A-Z][a-z]+)?)/i, groups: { name: 1 } },
    { regex: /([A-Z][a-z]+ ?([A-Z][a-z]+)?) from ([A-Za-z0-9\s&]+)/i, groups: { name: 1, company: 3 } },
  ];
  
  // Initialize with required properties
  let callerInfo: CallerInfo = { name: null, company: null, phone: null };
  
  for (const pattern of callerPatterns) {
    const match = transcript.match(pattern.regex);
    if (match) {
      if (pattern.groups.name && match[pattern.groups.name]) {
        callerInfo.name = match[pattern.groups.name].trim();
      }
      if (pattern.groups.company && match[pattern.groups.company]) {
        callerInfo.company = match[pattern.groups.company].trim();
      }
      break;
    }
  }
  
  // Extract contact information
  const phoneRegex = /(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}|extension\s+\d{3,4}|ext\.?\s+\d{3,4})/gi;
  const phoneMatches = [...transcript.matchAll(phoneRegex)];
  callerInfo.phone = phoneMatches.length > 0 ? phoneMatches.map(m => m[0]) : null;
  
  return callerInfo;
}

// Function to process voicemails and store in ChromaDB
async function processVoicemailsToChromaDB() {
  try {
    console.log("Creating/accessing ChromaDB collection for voicemails...");
    
    // Create or get the collection
    const collection = await chromaClient.getOrCreateCollection({
      name: "voicemail_transcripts",
    });
    
    console.log("Processing voicemail transcripts and generating embeddings...");
    
    // Process each voicemail
    for (const vm of voicemailTranscripts) {
      console.log(`\nProcessing Voicemail ${vm.id}...`);
      
      // Extract caller info
      const callerInfo = await extractCallerInfo(vm.transcript);
      
      // Generate embedding using Nomic Embed Text via Ollama
      const embedding = await getEmbedding(vm.transcript);
      
      if (!embedding) {
        console.error(`Failed to generate embedding for voicemail ${vm.id}. Skipping...`);
        continue;
      }
      
      // Create metadata compatible with ChromaDB requirements
      const metadata: ChromaMetadata = {
        from_name: callerInfo.name || "Unknown",
        from_company: callerInfo.company || "Unknown",
        phone: callerInfo.phone ? callerInfo.phone.join(", ") : "",
        timestamp: new Date().toISOString() // Replace with actual timestamp if available
      };
      
      // Add to ChromaDB - only if we have a valid embedding
      if (embedding) {
        await collection.upsert({
          ids: [vm.id],
          embeddings: [embedding],
          documents: [vm.transcript],
          metadatas: [metadata]
        });
        
        console.log(`Successfully added voicemail ${vm.id} to ChromaDB`);
      } else {
        console.log(`Skipping voicemail ${vm.id} due to missing embedding`);
      }
      
    }
    
    console.log("\nAll voicemails processed and stored in ChromaDB!");
    
    // Test query example
    const testQuery = "partnership agreement concerns";
    console.log(`\nTesting query: "${testQuery}"`);
    
    // Get embedding for the query
    const queryEmbedding = await getEmbedding(testQuery);
    
    // Search using the embedding (only if we have a valid embedding)
    if (queryEmbedding) {
      const results = await collection.query({
        queryEmbeddings: [queryEmbedding],
        nResults: 2,
      });
      
      console.log("Query results:");
      console.log(JSON.stringify(results, null, 5));
    } else {
      console.log("Could not generate embedding for test query");
    }
    
  } catch (error) {
    console.error("Error in voicemail processing:", error);
  }
}


async function main() {
  console.log("Voicemail Transcript Management System");
  console.log("======================================");
  console.log(`Using Ollama Host: ${OLLAMA_HOST}`);
  console.log(`Using ChromaDB Host: ${CHROMA_HOST}`);
  
  await processVoicemailsToChromaDB();
}

main().catch(console.error);

