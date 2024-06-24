import { VoyVectorStore } from "@langchain/community/vectorstores/voy";
import { WebPDFLoader } from "langchain/community/document_loaders/web/pdf";
import { RecursiveCharacterTextSplitter } from 'langchain/community/text_splitters/recursive_character';


// Function to load PDF from local file system, split text, generate embeddings, and store in VoyVectorStore
const loadAndSplitPDF = async (filePath) => {
  try {
    // Read the PDF file as a buffer
    const pdfBuffer = readFileSync(filePath);

    // Create a new Blob from the buffer
    const pdfBlob = new Blob([pdfBuffer], { type: 'application/pdf' });

    // Instantiate the WebPDFLoader with the Blob
    const loader = new WebPDFLoader(pdfBlob);

    // Load the PDF document
    const docs = await loader.load();

    // Instantiate the RecursiveCharacterTextSplitter
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: 1000, // Define your chunk size
      chunkOverlap: 200, // Define your chunk overlap
    });

    // Split the text from the loaded documents
    const splitDocs = splitter.splitDocuments(docs);
    console.log(splitDocs);
    // // Instantiate the OpenAIEmbeddings
    // const embeddings = new OpenAIEmbeddings({
    //   apiKey: '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', // Replace with your OpenAI API key
    // });

    // // Prepare documents with embeddings for storage
    // const documentsWithEmbeddings = await Promise.all(splitDocs.map(async (doc) => {
    //   const content = doc.content || ''; // Get the content of the split document
    //   const embedding = await embeddings.embedText(content); // Generate embeddings for the content
    //   return {
    //     content,
    //     embedding
    //   };
    // }));

    // Instantiate the VoyVectorStore
    const vectorStore = await VoyVectorStore.fromDocuments(splitDocs, embeddings, {
      indexName: 'pdf_documents', // Define your index name
    });

    console.log('Documents have been successfully processed and stored in the vector store.');
    return vectorStore;
  } catch (error) {
    console.error('Error processing PDF:', error);
  }
};

// Function to create a retriever for the stored documents
const createRetriever = async (vectorStore) => {
  try {
    const retriever = vectorStore.asRetriever();
    return retriever;
  } catch (error) {
    console.error('Error creating retriever:', error);
  }
};

// Path to your local PDF file
const pdfFilePath = "../data/Robinson Crusoe.pdf";

// Main function to execute the process
const langchainMain = async () => {
  const vectorStore = await loadAndSplitPDF(pdfFilePath);
  // const retriever = await createRetriever(vectorStore);

  // // Example query
  // const query = "What is the main theme of Robinson Crusoe?";
  // const results = await retriever.retrieve(query);

  // console.log('Search results:', results);
};


