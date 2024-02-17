from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI

ingredients_scheme = ResponseSchema(name="ingredients",
                                    description="What is the list of stuff I need before "
                                                "preparing the dish? what is the portion?"
                                                "מה הם המצרכים? מה הכמויות?")
instruction_scheme = ResponseSchema(name="instructions",
                                    description="What are the instructions for the recipe? what "
                                                "should be done? how should it be served? how soon? "
                                                "מה ההוראות הכנה? מה צריך לעשות? כמה מהר צריך להגיש?")
response_schemas = [ingredients_scheme,
                    instruction_scheme]

output_parser = StructuredOutputParser.from_response_schemas(response_schemas)

format_instructions = output_parser.get_format_instructions()

review_template = """\
For the following text, return only a JSON response with the following keys:

ingredients: What is the ingredient of the recipe? which items should be used? what is the portion?

instructions: What are the instructions for the recipe? what should be done? how will it be served? how soon?

1- Make sure the output stays in the original language.
2- Make sure the output is only JSON. nothing else!
3- Make sure ingredients has only list of ingredients not instruction for the recipe
 (nothing more, nothing less, do not miss anything!).
4- Make sure instructions has only list of instructions, which is everything the user needs to know and follow
from preparation, to cooking or baking, all the way to serving the dish, DO NOT MISS ANYTHING!
5- תשים לב שאתה לא שוכח כלום ברשימת ההוראות, כל מה שהמשתמש צריך בשביל להכין את המתכון,
 מההכנות לבישול או האפיה וכמובן כולל ההגשה! אל תשכח את ההגשה!

5- no links, no html, no nothing. just text as described above.


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
