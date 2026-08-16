import os
import sys
from google import genai

def main():
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable is not set.")
        sys.exit(1)
        
    print(f"Initializing Gemini client with provided API key...")
    # Initialize genai.Client with api_version="v1" to use stable endpoints
    client = genai.Client(api_key=api_key)
    
    try:
        print("Listing available models using client.models.list()...")
        models = list(client.models.list())
    except Exception as e:
        print(f"ERROR: Failed to list models: {e}")
        sys.exit(1)
        
    print("\nAll models available to this API key:")
    print("-" * 60)
    
    generate_content_models = []
    for m in models:
        actions = m.supported_actions or []
        # Check if generateContent is supported
        supports_gen = any("generateContent" in a or "generate_content" in a or "generate" in a.lower() for a in actions)
        
        status = "Supports generateContent" if supports_gen else "Other action"
        clean_name = m.name.replace("models/", "")
        print(f"- {clean_name:<30} | {status} | Actions: {actions}")
        
        if supports_gen:
            generate_content_models.append(clean_name)
            
    print("-" * 60)
    print(f"Found {len(generate_content_models)} models supporting generateContent.")
    
    if not generate_content_models:
        print("ERROR: No models supporting generateContent found.")
        sys.exit(1)
        
    # Choose default or select the best one
    default_model = "gemini-2.5-flash"
    if default_model in generate_content_models:
        selected = default_model
        print(f"\nSUCCESS: '{default_model}' is available and selected as default.")
    else:
        # Select the first available model that supports generateContent
        selected = generate_content_models[0]
        print(f"\nSUCCESS: '{default_model}' is NOT available. Selected alternative: '{selected}'")
        
    return selected

if __name__ == "__main__":
    main()
