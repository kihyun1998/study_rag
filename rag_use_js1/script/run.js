import { loadAndSplitPDF } from './run_langchain.js';


document.getElementById('myButton').addEventListener('click', () => {
    const pdfFilePath = "../data/무결성 기능.pdf";
    loadAndSplitPDF(pdfFilePath);
});
