# Daylivotion

A web application that generates daily Bible devotions using AI-powered verse selection from the Unlocked Dynamic Bible (UDB).

## Features

- **AI-Generated Verse Selection**: Uses llama-3-bible-dpo model to intelligently select the best verse for devotion
- **Random Verse Generation**: Automatically generates 9 random verses from the UDB Bible
- **Smart Selection**: AI picks the most suitable verse from 10 candidates (1 AI-generated + 9 random)
- **Web Interface**: Simple, clean web interface with a "Generate Devotion" button

## Requirements

- Python 3.7+
- llama-cli (llama.cpp command line tool)
- Vulkan support for GPU acceleration
- Unlocked Dynamic Bible (UDB) files in USFM format

## Setup

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Ensure llama-cli is installed and accessible in your PATH**

3. **Configure paths:**
   - Bible files should be at: `/media/novadev/storage/daylivotion/en_udb`
   - AI model should be at: `/media/novadev/storage/daylivotion/models/Meta-Llama-3.1-8B-Instruct-Q5_K_M.gguf`
   
   If your paths are different, update the default paths in:
   - `bible_parser.py` (line 7)
   - `ai_runner.py` (line 5)

## Running the Application

### Web Server

Start the Flask web server:

```bash
python app.py
```

The application will be available at `http://localhost:5100`

### Command Line

You can also run the devotion generator directly from the command line:

```bash
python devotion_generator.py
```

This will generate a devotion and print it to the terminal.

## How It Works

1. **AI Verse Generation**: The AI generates a random Bible verse reference
2. **Random Verse Generation**: 9 additional random verses are selected from the UDB Bible
3. **Content Retrieval**: The actual verse text is retrieved from the USFM Bible files
4. **AI Selection**: All 10 verses are sent to the AI, which selects the best one for devotion
5. **Output**: The selected verse is displayed (on web or terminal)

## File Structure

```
daylivotion/
├── app.py                    # Flask web application
├── bible_parser.py           # USFM Bible file parser
├── ai_runner.py              # llama-cli interface and output parser
├── devotion_generator.py     # Main devotion generation logic
├── requirements.txt          # Python dependencies
├── templates/
│   └── index.html           # Web interface
└── README.md                # This file
```

## Troubleshooting

- **llama-cli not found**: Ensure llama-cli is installed and in your PATH
- **Bible files not found**: Check that the path in `bible_parser.py` points to your UDB files
- **Model not found**: Check that the path in `ai_runner.py` points to your GGUF model file
- **Vulkan errors**: Ensure your GPU drivers and Vulkan runtime are properly installed

## License

This project uses the Unlocked Dynamic Bible, which is available under its own license terms.
