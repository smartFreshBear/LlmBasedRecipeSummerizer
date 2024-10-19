from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI


ingredients_scheme = ResponseSchema(name="ingredients",
                                    description="What is the list of stuff I need before "
                                                "Preparing the dish? what is the portion? "
                                                "What are the sub title of the ingredients"
                                                "מה הם המצרכים? מה הכמויות? מה הכותרות ביניים? תתי כותרות?")
instruction_scheme = ResponseSchema(name="instructions",
                                    description="What are the instructions for the recipe? what "
                                                "should be done? how should it be served? how soon? as String "
                                                "(ingridient name and portion of it)"
                                                "מה ההוראות הכנה? מה צריך לעשות? כמה מהר צריך להגיש?"
                                                "כטקסט שמכיל את השם של המרכיב ואת הכמות שלו")
response_schemas = [ingredients_scheme,
                    instruction_scheme]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()


review_template = """\
The following JSON, contains a recipe json with some noise and or/ corruption your task is to:

1- Clean it so it will include only recipe related, ingredients and instructions (likely the noise is only in the values)
2- Clean the text from any irrelevant information, symbols, or typos, urls, or any other non related text.
3- Just keep the relevant information of the cooking recipe.
4- Make sure all the sections (which looks like number, single letter, or a word) are kept in the output.
5- All recipe related data is there, BUT text which is not related to the recipe, remove it.
6- DO NOT remove any additional instructions, decoration, serving details, or any other text that might be 
important for serving the dish or any peripheral information that might support the recipe.
כל מה שקשור להגשה, לדרך שבה חותכים, מקררים, מחממים זה גם חלק מההוראות וצריך להישאר בתוצאה.
7- Of course the output is only JSON, and the text is a string value for each key (no inner keys),
 and the value should stay in the original language.
8- Urls, encoded related stuff like 9c%d7% or 
%d7%a7%d7%9c other not relevant symbols or special that has no reason to appear (\\ * % # @ etc) - remove them.
9- Eventually, all outputs that are processed by you should be align, to be aligned to one simple format.
10- Return a (ONLY !) JSON response with the following (only) keys, 1- ingredients, 2- instructions.

ingredients: contains the list of ingredients (list of text only, line per ingredient) and
 their sub category (usually comes with number that make the stage) or headline, 
 and the portion of each ingredient, no inner keys should be included, just the 
 ingredient name and portion as text (avoid having inner "ingredients" or מרכיבים inside the text),
  if the portion is missing, estimate it given the instructions, it can be grams, cups, spoons,
   or any other unit, DO NOT MISS ANYTHING!

instructions: contains list of instructions (list of text, line per instruction)
which is everything the user needs to know and follow from preparation, to cooking or baking, all the way to serving the dish, DO NOT MISS ANYTHING!

We don't want anything else but a json response with the above keys, and the text as a string value for each key.
otherwise it won't be parsed correctly.

text: {text}

{format_instructions}
"""


# The following class is a service that is used to summarize recipes
class RecipeSummarizer:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo")
        self.prompt = ChatPromptTemplate.from_template(template=review_template)

    def summarize_recipe(self, recipe: str) -> dict:
        messages = self.prompt.format_messages(text=recipe, format_instructions=format_instructions)
        response = self.llm.invoke(messages)
        print(f'response: {response}')
        result = output_parser.parse(response.content)
        print(result)
        return result
