"""Quick test script to verify Vertex AI is working"""
import os
from dotenv import load_dotenv
load_dotenv()
import vertexai
from vertexai.preview.generative_models import GenerativeModel

project_id = os.getenv('GCP_PROJECT_ID')
print(f"Testing Vertex AI with project: {project_id}")

try:
    vertexai.init(project=project_id, location='us-central1')
    print("✓ Vertex AI initialized")
    
    model = GenerativeModel('gemini-pro')
    print("✓ Model created")
    
    response = model.generate_content('Say hello in one word')
    print(f"✓ Success! Response: {response.text}")
    print("\n🎉 Vertex AI is working correctly!")
except Exception as e:
    print(f"✗ Error: {e}")
    print("\nPlease check:")
    print("1. Vertex AI API is enabled in Google Cloud Console")
    print("2. Service account has 'Vertex AI User' role")
    print("3. Project ID is correct")
