# Environment Setup

## Required Environment Variables

This project requires an OpenAI API key to function properly.

### Setup Instructions:

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_actual_openai_api_key_here
   ```

3. Install python-dotenv if not already installed:
   ```bash
   pip install python-dotenv
   ```

### Important Notes:

- Never commit the `.env` file to version control
- The `.env` file is already included in `.gitignore`
- Use `.env.example` as a template for required variables
- Keep your API keys secure and never share them publicly

### Getting an OpenAI API Key:

1. Visit https://platform.openai.com/
2. Sign up or log in to your account
3. Navigate to API Keys section
4. Create a new API key
5. Copy the key and add it to your `.env` file