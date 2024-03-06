# Use the official Python image for 3.11 as the base image
FROM python:3.11.6

# Set the working directory inside the container
WORKDIR /app

# Copy only the necessary files for poetry installation
COPY pyproject.toml poetry.lock /app/

# Install poetry and project dependencies
RUN pip install poetry \
        && poetry config virtualenvs.create false \
    && poetry install

# Copy the rest of the application code
COPY llm_based_recipe_summerizer /app/llm_based_recipe_summerizer/

# Expose the port that FastAPI will run on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "llm_based_recipe_summerizer.controller:app", "--host", "0.0.0.0", "--port", "8000"]

