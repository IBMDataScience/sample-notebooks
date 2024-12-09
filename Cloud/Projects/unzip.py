import zipfile
import os

def unzip_all(zip_folder, extract_to):
    # Ensure the extract_to folder exists
    os.makedirs(extract_to, exist_ok=True)

    # Check if the zip_folder exists and contains files
    if not os.path.isdir(zip_folder):
        print(f"Error: The specified folder '{zip_folder}' does not exist.")
        return

    # Get a list of all .zip files in the directory
    zip_files = [f for f in os.listdir(zip_folder) if f.endswith(".zip")]
    if not zip_files:
        print("No .zip files found in the specified folder.")
        return

    # Iterate through each zip file and extract it
    for filename in zip_files:
        zip_path = os.path.join(zip_folder, filename)

        # Ensure it's a valid zip file
        if zipfile.is_zipfile(zip_path):
            try:
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    print(f"Extracting '{filename}' to '{extract_to}'...")
                    zip_ref.extractall(extract_to)
                    print(f"'{filename}' extracted successfully.")
            except Exception as e:
                print(f"Failed to extract '{filename}': {e}")
        else:
            print(f"'{filename}' is not a valid zip file.")

# Example usage:
zip_folder = "/path/to/your/zip/folder"
extract_to = "/path/to/extract/destination"

unzip_all(zip_folder, extract_to)
