def prompt(context_chunks, query):
    context = "\n\n".join(context_chunks)
    return f"""Use the following context to answer the question.
    
    Context: {context}
    Question: {query}
    Answer:"""