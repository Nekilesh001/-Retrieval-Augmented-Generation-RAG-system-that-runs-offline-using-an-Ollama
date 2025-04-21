import requests

def ask_ollama(query, model="llama2"):
    prompt = f"""Answer the following question based on the model's general knowledge.

Question: {query}

Answer:"""

    # Send the request to the Llama model API
    response = requests.post(
        "http://localhost:11434/api/generate",  # Replace with your model's API endpoint
        json={"model": model, "prompt": prompt}
    )

    # Print the raw response content for debugging
    print("Raw response content: ", response.content)

    if response.status_code == 200:
        try:
            # You can manually clean the response if it contains extra data before JSON parsing
            response_text = response.content.decode('utf-8')
            # Optional: If there's any unexpected extra data, trim it or process accordingly
            print("Processed response text: ", response_text)
            return response.json()["response"]
        except ValueError as e:
            print("Error parsing JSON:", e)
            return "Error: Unable to parse the response."
    else:
        return f"Error: {response.status_code} - {response.text}"
