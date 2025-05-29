import json

def merge_files_to_jsonl(src_file_path, tgt_file_path, output_file_path):
    """
    Merges two text files line by line into a JSONL file.

    Each line in the output file will be a JSON object with 'src' and 'trg' keys.

    Args:
        src_file_path (str): Path to the source text file (for 'src' field).
        tgt_file_path (str): Path to the target text file (for 'trg' field).
        output_file_path (str): Path for the output JSONL file.
    """
    try:
        with open(src_file_path, 'r', encoding='utf-8') as src_file, \
             open(tgt_file_path, 'r', encoding='utf-8') as tgt_file, \
             open(output_file_path, 'w', encoding='utf-8') as outfile:

            for src_line, tgt_line in zip(src_file, tgt_file):
                # Strip newline characters from the end of each line
                src_content = src_line.strip()
                tgt_content = tgt_line.strip()

                # Create the JSON object
                data = {"src": src_content, "trg": tgt_content}

                # Write the JSON object as a string to the output file, followed by a newline
                outfile.write(json.dumps(data) + '\n')
        
        print(f"Successfully merged files into: {output_file_path}")

    except FileNotFoundError:
        print("Error: One or both input files were not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == '__main__':
    # Define the paths to your input files
    src_file = "/home/iiserver32/Workbench/fyb/controllable-diffusion/baselines/latent-diffusion-for-language/datasets/CommonGen/commongen.test.src_alpha.txt"
    tgt_file = "/home/iiserver32/Workbench/fyb/controllable-diffusion/baselines/latent-diffusion-for-language/datasets/CommonGen/commongen.test.tgt.txt"
    
    # Define the path for your output JSONL file
    # It will be created in the same directory as the input CommonGen files
    output_file = "/home/iiserver32/Workbench/fyb/controllable-diffusion/baselines/latent-diffusion-for-language/datasets/CommonGen/test.jsonl"

    merge_files_to_jsonl(src_file, tgt_file, output_file)
