import typer
from pathlib import Path
from .main import process_legal_document

app = typer.Typer()

def get_scenario_from_input() -> str:
    print("\nWhat would you like to analyze in this legal document?")
    print("Type your analysis requirements below (press Enter twice when done):\n")
    
    lines = []
    while True:
        line = input()
        if line:
            lines.append(line)
        elif lines:
            break
            
    scenario = "\n".join(lines)
    return scenario
    
@app.command()
def analyze(pdf_path: Path = typer.Argument(..., help="Path to the PDF file")):
    print(f"Analyzing PDF: {pdf_path}")
    if not pdf_path.exists():
        typer.echo(f"Error: File {pdf_path} not found")
        raise typer.Exit(1)
    
    try:
        scenario = get_scenario_from_input()
        result = process_legal_document(pdf_path, scenario)
        typer.echo(result)
    except Exception as e:
        typer.echo(f"Error: {str(e)}")
        raise typer.Exit(1)

if __name__ == "__main__":
    app()
