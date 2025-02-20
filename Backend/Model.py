import cohere  # Import cohere library for AI services
from rich import print  # Import the rich library to enhance terminal outputs
from dotenv import dotenv_values  # Import dotenv to load environmental variables from a .env file

# Load environment variables from the .env file
env_vars = dotenv_values(".env")

# Retrieve API Key
CohereAPIKey = env_vars.get("CohereAPIKey")

# Validate the Cohere API key
if not CohereAPIKey:
    print("[bold red]Error:[/bold red] Cohere API key not found in .env file. Please add your API key as 'CohereAPIKey'.")
    exit()

# Create a cohere client using the provided API key
co = cohere.Client(api_key=CohereAPIKey)

# Define the preamble that guides the AI model on how to categorize queries
preamble = """
You are a Decision-Making Model that determines if a query needs realtime data or can be answered with general knowledge.

-> Respond with 'realtime (query)' if the query requires current information like:
   - Current events, news, or updates
   - Present status of people or things
   - Latest information that may change over time

-> Respond with 'general (query)' for:
   - General knowledge questions
   - Personal opinions or advice
   - Historical information
   - Emotional or conversational exchanges
"""

# Define the function for processing user queries
def FirstLayerDMM(query: str) -> list:
    try:
        response = co.generate(
            model="command-xlarge-nightly",
            prompt=f"{preamble}\nUser: {query}\nDecision:",
            max_tokens=128,
            temperature=0.7,
            stop_sequences=["\n"]
        )
        decision = response.generations[0].text.strip()
        return [decision if decision else f"general {query}"]
    except Exception as e:
        return [f"general {query}"]

# Entry point for the script
if __name__ == "__main__":
    print("[bold cyan]Welcome to the Decision-Making Model![/bold cyan]")
    print("[bold yellow]Type 'quit' or 'exit' to stop.[/bold yellow]")

    # Continuously prompt the user for input and process it
    while True:
        user_input = input(">>> ")
        if user_input.lower() in ["quit", "exit"]:
            print("[bold green]Goodbye![/bold green]")
            break
        result = FirstLayerDMM(user_input)
        print(f"[bold blue]Decision:[/bold blue] {result}")
