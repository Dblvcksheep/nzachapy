from sementic import embed, match_embeddings

class Nzacha:
    def __init__(self,
                 chunk_size = 200,
                 overlap = 20,
                 openai_api_key = None,
                 threshold = 0.6):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.openai_api_key =openai_api_key
        self.threshold = threshold

        self.text_chunks = []
        self.embeddings = []

    def add_by_words(self, data):
        data = data.split()

        if self.overlap >= self.chunk_size:
            raise ValueError("Overlap must be smaller than chunk size")
        start = 0
        while start < len(data):
            end = start + self.chunk_size
            chunk = data[start:end]

            chunk_text = " ".join(chunk)

            self.text_chunks.append(chunk_text)

            embedding = embed(chunk_text, self.openai_api_key)
            self.embeddings.append(embedding)

            start += (self.chunk_size - self.overlap)

        return self.text_chunks

    def search(self, query):
        query_embedding = embed(query, self.openai_api_key)

        related = match_embeddings(self.embeddings, query_embedding, self.threshold)

        related_data = []

        for item in related:
            index = item["index"]
            related_data.append(self.text_chunks[index])

        return " ".join(related_data)




