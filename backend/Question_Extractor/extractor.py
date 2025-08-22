from Question_Extractor.src.pdf_processing import load_pdf_with_fitz
from Question_Extractor.src.llm_client import LocalLLMClient
from Question_Extractor.src.pattern_extractor import EnhancedPatternExtractor
from Question_Extractor.src.question_processor import process_single_pdf_with_verification
from Question_Extractor.convert_doc_to_pdf import convert_all_docx_in_test_files #check if this works

def main():
    """Main entry point for the question extraction pipeline"""
    output = process_single_pdf_with_verification(model="ibnzterrell/Meta-Llama-3.3-70B-Instruct-AWQ-INT4", 
                                     base_url="http://localhost:9091/v1")
    
    print("$"*50)
    print("the extracted string")
    print("$"*50)
    print(output)

if __name__ == "__main__":
    main()