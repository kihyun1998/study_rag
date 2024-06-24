import { WebPDFLoader } from "@langchain/community/document_loaders/web/pdf";

import { RecursiveCharacterTextSplitter } from '@langchain/community/text_splitters/recursive_character';
// Define an async function to load and process the PDF
async function loadAndSplitPDF(filePath) {
  // Create an instance of PDFLoader with the file path
  const pdfLoader = new WebPDFLoader(filePath);

  // Load the PDF content
  const pdfContent = await pdfLoader.load();

  // Create an instance of RecursiveCharacterTextSplitter with the desired options
  const textSplitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000, // Maximum chunk size
    chunkOverlap: 100, // Overlap between chunks
  });

  // Split the loaded PDF content into chunks
  const chunks = textSplitter.split(pdfContent);

  // Return the chunks
  return chunks;
}

export { loadAndSplitPDF };

