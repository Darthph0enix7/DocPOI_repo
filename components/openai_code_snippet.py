import openai

openai.api_key = 'your-api-keydsad'

response = openai.Completion.create(
  engine="davinci",
  prompt="Hello, world!",
  max_tokens=5
)

print(response.choices[0].text.strip())