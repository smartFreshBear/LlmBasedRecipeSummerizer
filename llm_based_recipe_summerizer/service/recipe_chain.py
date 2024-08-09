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
The following text, contains a recipe, your task is to:

1- Extract the ingredients and the instructions from the text.
1.1 Mind that instructions are a bit tricky to extract, they are usually the steps that the user needs to follow to prepare the dish.
1.2 Because of the instructions challange we prefer to just keep text even if it is not 100% related to the instructions, just to be on the safe side.
2- Summarize the recipe, but keep all the essential information (be careful not to drop any stage/phase/section or text which might be important for the recipe).
2.1 While Summering do not drop any essential information (if you are not sure, keep it!!!)
2.2 Sometimes the text is already short and well summarized, so the only thing you need to do is step 3.
2.3 Usually parenthesis () are very essential for the recipe, so keep them in the output!!!
2.4 While summarizing, be super aware of numbers and units, they are very important for the recipe, so text with numbers (can be units, or just number represinting stages in the instructions) should be kept in the output.
2.5 Make sure the numbers has at least one space before and after them, so they are not attached to the text.
3. make sure not to return just a string of char but rather words that make sense and are related to the recipe.
4- Keep all the process relevant for the cooking process, from preparation, to cooking or baking, all the way to serving the dish.
4.1 make the response as long as it's need to be to keep all the relevant information.
4.2 Sometimes the text contained some symbols like #, * or any other that are not related to the recipe, make sure to remove them.
4.3 If the text has some typo, make sure to correct it, but do not change the meaning of the text.
5- return a (only!) JSON response with the following (only) keys:
 
ingredients: contains the list of ingredients (list of text only, line per ingredient) and their sub category (usually comes with number that make the stage) or headline, and the portion of each ingredient, no inner keys should be included, just the ingredient name and portion as text (avoid having inner "ingridentns" or מרכיבים inside the text), if the portion is missing, estimate it given the instructions, it can be grams, cups, spoons, or any other unit, DO NOT MISS ANYTHING!
instructions: contains list of instructions (list of text, line per instruction) , which is everything the user needs to know and follow from preparation, to cooking or baking, all the way to serving the dish, DO NOT MISS ANYTHING!

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
