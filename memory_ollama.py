import chromadb
import ollama

class MemoryManager:
    def __init__(self, model_name="mxbai-embed-large"):
        self.client = chromadb.Client()
        self.model_name = model_name

        # Check if the collection already exists
        existing_collections = self.client.list_collections()
        if any(collection.name == "memory" for collection in existing_collections):
            self.collection = self.client.get_collection(name="memory")
        else:
            self.collection = self.client.create_collection(name="memory")

    def save_memory(self, memory_id, content):
        response = ollama.embeddings(model=self.model_name, prompt=content)
        embedding = response["embedding"]
        self.collection.add(
            ids=[memory_id],
            embeddings=[embedding],
            documents=[content]
        )

    def retrieve_memory(self, query, top_results=1):
        response = ollama.embeddings(prompt=query, model=self.model_name)
        query_embedding = response["embedding"]
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_results
        )
        retrieved_memory = results['documents'][0][0]
        return retrieved_memory

    def generate_response(self, prompt, memory):
        output = ollama.generate(
            model="qwen:0.5b",
            prompt=f"Using this memory: {memory}. Respond to this prompt: {prompt}"
        )
        return output['response']