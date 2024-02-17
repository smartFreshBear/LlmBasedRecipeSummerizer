import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from llm_based_recipe_summerizer.service.recipe_chain import RecipeSummarizer

app = FastAPI()

recipe_summarizer = RecipeSummarizer()


class RawText(BaseModel):
    text: str


@app.get("/summarize_recipe/")
def get_summarize_recipe(raw_text: RawText) -> dict:
    print(f'received text: {raw_text.text}')
    return recipe_summarizer.summarize_recipe(raw_text.text)


@app.get("/")
def root():
    return 'go fuck yourself! :)'


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)