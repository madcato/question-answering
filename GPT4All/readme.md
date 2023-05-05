# GPT4All Setup: Easy Peasy

The setup was the easiest one. Their [Github instructions](https://github.com/nomic-ai/gpt4all) are well-defined and straightforward. There are two options, local or google collab. I tried both and could run it on my M1 mac and google collab within a few minutes.

## Local Setup

- Download the `gpt4all-lora-quantized.bin` file from [Direct Link](https://the-eye.eu/public/AI/models/nomic-ai/gpt4all/gpt4all-lora-quantized.bin).
- Clone [this repository](https://github.com/nomic-ai/gpt4all), navigate to `chat`, and place the downloaded file there.
- Run the appropriate command for your OS:
- M1 Mac/OSX: `cd chat;./gpt4all-lora-quantized-OSX-m1`
- Linux: `cd chat;./gpt4all-lora-quantized-linux-x86`
- Windows (PowerShell): `cd chat;./gpt4all-lora-quantized-win64.exe`
- Intel Mac/OSX: `cd chat;./gpt4all-lora-quantized-OSX-intel`

