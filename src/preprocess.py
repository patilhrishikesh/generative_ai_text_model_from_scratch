def load_and_clean_text(file_path):
    
    # read a text file and performs basic preprocessing
    with open(file_path, "r", encoding="utf-8") as file:
        text = file.read().lower()
        
    return text