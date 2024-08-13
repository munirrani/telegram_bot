from functools import wraps
import json
import sys
from openai import OpenAI

import os
import logging
import copy
import tiktoken
from telegram.constants import ParseMode, ChatAction
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    ContextTypes,
    CommandHandler,
    MessageHandler,
    filters,
)
from notion_client import Client, AsyncClient
from enum import Enum
import html
import traceback
import base64
from io import BytesIO
from PIL import Image
import requests

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

telegram_bot_key = os.environ["TELEGRAM_BOT"]
notion_token = os.environ["NOTION_TOKEN"]
notion = Client(auth=notion_token, log_level=logging.ERROR)

conversation_state = {}
CONVERSATION_STATE_FILE = "conversation_state.json"

logging.basicConfig(format="%(message)s", level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAMES = {
    # "GPT-3.5": "gpt-3.5-turbo-0125",
    "GPT-4O": "gpt-4o",
    "GPT-4O-Mini": "gpt-4o-mini",
    "GPT-4": "gpt-4-turbo",
    "Mistral Small": "mistral-small",
    "Mistral Medium": "mistral-medium",
    "Dall-E 3": "dall-e-3",
}


def custom_print(*args, **kwargs):
    output = " ".join(str(arg) for arg in args)
    sys.stdout.write(output + "\n")
    sys.stdout.flush()


# Override the print() function with our custom_print() function
print = custom_print


def get_chat_bot_message_template(update: Update):
    first_name = (
        update.message.from_user.first_name
        if update.message.from_user.first_name
        else ""
    )
    last_name = (
        update.message.from_user.last_name if update.message.from_user.last_name else ""
    )
    name = first_name + " " + last_name

    system_prompt = [
        {
            "role": "system",
            "content": f"You've been programmed to be put inside Telegram chat to become a chat assistant for {name}. Each person can speak English and Malay Bahasa Melayu fluently. Answer in a very casual manner, like a close friend. Use emojis. Use emojis. Make jokes and puns. Be witty and creative. Please answer directly.",
        }
    ]

    return system_prompt


def num_tokens_from_messages(messages, model=MODEL_NAMES["GPT-4O-Mini"]):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    if model in list(MODEL_NAMES.values()):
        tokens_per_message = 3
        tokens_per_name = 1
    else:
        raise NotImplementedError(
            f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
        )
    num_tokens = 0
    if len(messages) == 0:
        return num_tokens
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>
    return num_tokens


async def get_llm_response(
    model_name, message_array, temperature, base64_encoded_string=None
):
    reply = ""
    try:
        if base64_encoded_string is not None:
            # make a copy of message_array
            message_array_temp = copy.deepcopy(message_array)
            message_array_temp[len(message_array_temp) - 1]["content"] = [
                {
                    "type": "text",
                    "text": message_array_temp[len(message_array_temp) - 1]["content"],
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,${base64_encoded_string}",
                    },
                },
            ]
            chat = client.chat.completions.create(
                model=model_name,
                messages=message_array_temp,
                temperature=temperature,
                max_tokens=2000,
                # presence_penalty=1.5,
            )
        elif model_name.startswith("mistral"):
            r = requests.post(
                url="https://api.mistral.ai/v1/chat/completions",
                headers={"Authorization": f'Bearer {os.environ["MISTRAL_API_KEY"]}'},
                json={
                    "model": model_name,
                    "messages": message_array,
                    "temperature": temperature - 0.2,
                    # mistral temp caps at 1, OpenAI meanwhile at 2
                },
            )
            chat = r.json()
        else:
            chat = client.chat.completions.create(
                model=model_name,
                messages=message_array,
                temperature=temperature,
                max_tokens=4000,
                # presence_penalty=1,
            )

        if model_name.startswith("gpt"):
            model = chat.model
            reply = chat.choices[0].message.content
            usage = {
                "prompt_tokens": chat.usage.prompt_tokens,
                "completion_tokens": chat.usage.completion_tokens,
                "total_tokens": chat.usage.total_tokens,
            }
        else:  # mistral
            model = chat["model"]
            reply = chat["choices"][0]["message"]["content"]
            usage = chat["usage"]

        return {
            "model": model,
            "reply": reply,
            "usage": usage,
        }
    except Exception as e:
        print("Error:", e)
        return {
            "reply": "Error in ChatGPT:\n\n"
            + e["response"]["data"]["error"]["message"],
            "usage": {},
            "model": model_name,
        }


def check_messages_size(model_name, prompt, selected_array, max_tokens=14384):
    completion_token_quota = (
        2500  # change from 1500 to 2500 to make it remember context
    )

    # Perform a deep copy of the message array
    selected_array_temp = selected_array.copy()

    index_to_pop_from = 1  # 0 is the system message

    while (
        num_tokens_from_messages(selected_array_temp, model_name)
        > (max_tokens - completion_token_quota)
        and len(selected_array_temp) > index_to_pop_from
    ):
        print(f"Popping from index {index_to_pop_from} due to max tokens")
        selected_array_temp.pop(index_to_pop_from)

    return {
        "model_name": model_name,
        "selected_array": selected_array_temp,
    }


async def perform_llm_completion(
    update: Update, model_name, prompt, base64_encoded_string=None
):
    global conversation_state
    username = update.message.from_user.username

    if username not in conversation_state:
        conversation_state[username] = {}

    conversation_state_user = conversation_state[username]

    if "chat_bot_message" not in conversation_state_user:
        conversation_state_user["chat_bot_message"] = get_chat_bot_message_template(
            update
        )

    if "chatgpt_temperature" not in conversation_state_user:
        conversation_state_user["chatgpt_temperature"] = 1.1

    conversation_state_user["last_gpt_model_used"] = model_name
    conversation_state_user["chat_bot_message"].append(
        {"role": "user", "content": prompt}
    )

    check_messages = check_messages_size(
        model_name, prompt, conversation_state_user["chat_bot_message"]
    )
    conversation_state_user["chat_bot_message"] = check_messages["selected_array"]

    response = await get_llm_response(
        check_messages["model_name"],
        conversation_state_user["chat_bot_message"],
        conversation_state_user["chatgpt_temperature"],
        base64_encoded_string,
    )
    reply = response["reply"]
    model = response["model"]
    usage = response["usage"]
    print(reply)  # In case the formatting is wrong

    # get the price
    try:
        cost_request = requests.post(
            url="http://127.0.0.1:8000/count-token-cost/",
            json={
                "llm_model_used": model_name,
                "usage": usage,
            },
        )
        token_cost = json.loads(cost_request.text)
    except Exception as e:
        print("Error calling localhost:8000/count-token-cost:", e)
        token_cost = {}

    conversation_state_user["chat_bot_message"].append(
        {"role": "assistant", "content": reply}
    )

    return {
        "reply": reply,
        "model": model,
        "usage": usage,
        "prompt": prompt,
        "token_cost": token_cost,
    }


async def perform_dalle(model_name, prompt):
    image_dimension = "1024x1024"
    quality = "standard"
    try:
        response = client.images.generate(
            model=model_name, prompt=prompt, n=1, size=image_dimension, quality=quality
        )
        print(f"DALL-E main response: {response}")
        image_url = response.data[0].url
    except Exception as e:
        print("Error in DALL-E:", e)
        image_url = ""
        return "Error in DALL-E", {
            "cost_in_original_currency": 0,
            "cost_in_myr": 0,
            "currency": "USD",
        }

    try:
        cost_response = requests.post(
            url="http://127.0.0.1:8000/count-dalle-cost/",
            json={
                "image_dimension": image_dimension,
                "image_quality": quality,
            },
        )
    except Exception as e:
        print("Error calling localhost:8000/count-dalle-cost:", e)
        cost = {}
    cost = json.loads(cost_response.text)
    return image_url, cost


def sanitize_text_for_telegram_bot(text: str):
    return text.replace("_", "\\_").replace("*", "\\*").replace("`", "\\`")


def combine_metadata_infos(metadata: dict):
    result = ""
    result += metadata["source"]
    # check if "date" is in metadata
    if "date" in metadata:
        result += f" ({metadata['date']})"
    return result


async def perform_chatgpt_langchain_query_completion(
    update: Update,
    query: str,
    model_name: str,
    temperature: float,
    system_prompt: str,
    search_count: int,
):
    # query, model, temperature, system_prompt, search_count
    # Do request to 127.0.0.1:8000/query-document/ POST
    try:
        request = requests.post(
            url="http://127.0.0.1:8000/query-document/",
            json={
                "query": query,
                "model": model_name,
                "temperature": temperature,
                "system_prompt": system_prompt,
                "search_count": search_count,
            },
        )
    except Exception as e:
        print("Error calling localhost:8000/query-document:", e)
        return "Error in ChatGPT LangChain query completion"

    result = json.loads(request.text)
    answer = result["answer"]
    source_documents = result["source_documents"]

    global conversation_state
    conversation_state_user = conversation_state[update.message.from_user.username]
    conversation_state_user["chat_bot_message"].append(
        {"role": "user", "content": query}
    )
    conversation_state_user["chat_bot_message"].append(
        {"role": "assistant", "content": answer}
    )

    code_snippet = f"{answer}\n\n*Sources:*\n\n"

    source_documents_list = [
        "\[{}] {}: {}".format(
            index + 1,
            combine_metadata_infos(document["metadata"]),
            sanitize_text_for_telegram_bot(document["page_content"]),
        )
        for index, document in enumerate(source_documents)
    ]

    result = code_snippet + "\n".join(source_documents_list)
    return result
    # return answer


def send_typing_action(func):
    """Sends typing action while processing func command."""

    @wraps(func)
    async def command_func(update, context, *args, **kwargs):
        await context.bot.send_chat_action(
            chat_id=update.effective_message.chat_id, action=ChatAction.TYPING
        )
        return await func(update, context, *args, **kwargs)

    return command_func


async def send_possibly_long_message(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    message: str,
    parse_mode: ParseMode = ParseMode.MARKDOWN,
):
    message = message.replace("```python", "```").replace("```javascript", "```")
    # save the latest message in a text file
    with open("latest_response.txt", "w") as f:
        f.write(message)
    try:
        if len(message) > 4096:
            for x in range(0, len(message), 4096):
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    text=message[x : x + 4096],
                    parse_mode=parse_mode,
                )
        else:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=message,
                parse_mode=parse_mode,
            )
    except Exception as e:
        print("Error:", e)


async def send_message_with_cost(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    reply_message: str,
    token_cost: dict,
):
    total_in_original_currency = (
        token_cost["prompt_cost_in_original_currency"]
        + token_cost["completion_cost_in_original_currency"]
    )
    total_in_myr = (
        token_cost["prompt_cost_in_myr"] + token_cost["completion_cost_in_myr"]
    )
    token_message = f"Tokens costed {round(total_in_original_currency, 5)} {token_cost['currency']} ({round(total_in_myr, 5)} MYR)"
    # first_message
    await send_possibly_long_message(update, context, reply_message)
    # second_message
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=token_message,
        parse_mode=ParseMode.MARKDOWN,
    )


@send_typing_action
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pass
    # Get the quoted message
    # quoted_message = update.message.reply_to_message
    # print(quoted_message)
    # quoted_caption = quoted_message.caption
    # quoted_image = quoted_message.photo[-1]
    # file_id = quoted_image.file_id
    # file = await context.bot.get_file(file_id)
    # file_path = file.file_path

    # timestamp = int(update.message.date.timestamp())
    # print(f"Time of message: {timestamp}")
    completion = await perform_llm_completion(
        update, MODEL_NAMES["GPT-4O-Mini"], update.message.text, None
    )
    log_events(update, completion)
    save()
    await send_message_with_cost(
        update, context, completion["reply"], completion["token_cost"]
    )


@send_typing_action
async def handle_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pass
    file_id = update.message.photo[-1].file_id

    # Allow quoted message to be used including images

    # Get the quoted message
    # quoted_message = update.message.reply_to_message

    # Get the image in the quoted message
    # quoted_image = quoted_message.photo[-1]

    # print(quoted_image)

    # Get the file path
    file = await context.bot.get_file(file_id)
    file_path = file.file_path

    # Download the file using requests
    response = requests.get(file_path)

    # Check if the download was successful
    if response.status_code == 200:
        # Convert the response content to an Image
        image = Image.open(BytesIO(response.content))

        # Convert and encode the image to base64
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")

        caption = update.message.caption

        if update.message.caption == None:
            caption = ""

        # Send the encoded image to the API
        completion = await perform_llm_completion(
            update, MODEL_NAMES["GPT-4"], caption, encoded_string
        )
        log_events(update, completion)

        await send_message_with_cost(
            update, context, completion["reply"], completion["token_cost"]
        )
    else:
        await context.bot.send_message(
            chat_id=update.effective_chat.id, text="Error in downloading image"
        )


@send_typing_action
async def handle_gpt4_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pass
    context.args.pop(0)
    argument = " ".join(context.args)
    completion = await perform_llm_completion(
        update, MODEL_NAMES["GPT-4O"], argument, None
    )
    log_events(update, completion)
    save()
    await send_message_with_cost(
        update, context, completion["reply"], completion["token_cost"]
    )


@send_typing_action
async def handle_mistral_small_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
):
    pass
    context.args.pop(0)
    argument = " ".join(context.args)
    completion = await perform_llm_completion(
        update, MODEL_NAMES["Mistral Small"], argument, None
    )
    log_events(update, completion)
    save()
    await send_message_with_cost(
        update, context, completion["reply"], completion["token_cost"]
    )


@send_typing_action
async def handle_mistral_medium_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
):
    pass
    context.args.pop(0)
    argument = " ".join(context.args)
    completion = await perform_llm_completion(
        update, MODEL_NAMES["Mistral Medium"], argument, None
    )
    log_events(update, completion)
    save()
    await send_message_with_cost(
        update, context, completion["reply"], completion["token_cost"]
    )


@send_typing_action
async def handle_bot_prompt_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pass
    global conversation_state
    conversation_state_user = conversation_state[update.message.from_user.username]

    argument = " ".join(context.args)
    if argument == "":
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=conversation_state_user["chat_bot_message"][0]["content"],
        )
        return
    else:
        conversation_state_user["chat_bot_message"][0]["content"] = argument
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"Bot prompt updated!",
        )


@send_typing_action
async def handle_unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pass
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="Sorry, I didn't understand that command.",
    )


async def handle_reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global conversation_state

    conversation_state_user = conversation_state[update.message.from_user.username]
    conversation_state_user["chat_bot_message"] = [
        conversation_state_user["chat_bot_message"][0]
    ]
    save()
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f"Conversation resetted!",
    )


async def handle_hard_reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global conversation_state

    conversation_state_user = conversation_state[update.message.from_user.username]
    conversation_state_user["chat_bot_message"] = get_chat_bot_message_template(update)
    conversation_state_user["last_gpt_model_used"] = MODEL_NAMES["GPT-4O-Mini"]
    conversation_state_user["chatgpt_temperature"] = 1.1
    save()
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=f"Conversation resetted!",
    )


async def handle_temp_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global conversation_state
    conversation_state_user = conversation_state[update.message.from_user.username]

    if " ".join(context.args) == "":
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"Temperature is {conversation_state_user['chatgpt_temperature']} for ChatGPT.",
        )
        return
    else:
        argument = context.args[0]
        if argument.replace(".", "").isnumeric():
            argument = float(argument)
            conversation_state_user["chatgpt_temperature"] = float(argument)
        else:
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=f"Invalid argument!",
            )
            return
        save()
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=f"Temperature set to {float(argument)}",
        )


@send_typing_action
async def handle_regen_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pass

    conversation_state_user = conversation_state[update.message.from_user.username]

    conversation_state_user[
        "chat_bot_message"
    ].pop()  # this is the "system" role response
    prompt = conversation_state_user["chat_bot_message"].pop()["content"]

    completion = await perform_llm_completion(
        update, conversation_state_user["last_gpt_model_used"], prompt, None
    )
    log_events(update, completion)
    save()
    await send_message_with_cost(
        update, context, completion["reply"], completion["token_cost"]
    )


@send_typing_action
async def handle_dalle_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    pass
    context.args.pop(0)
    prompt = " ".join(context.args)
    image_url, cost = await perform_dalle(MODEL_NAMES["Dall-E 3"], prompt)
    if image_url[:5] == "Error":
        await context.bot.send_message(
            chat_id=update.effective_chat.id, text="Error in DALL-E"
        )
    else:
        await context.bot.send_photo(chat_id=update.effective_chat.id, photo=image_url)
        cost_text = f"Costed {round(cost['cost_in_original_currency'], 5)} {cost['currency']} ({round(cost['cost_in_myr'], 5)} MYR)"
        await context.bot.send_message(chat_id=update.effective_chat.id, text=cost_text)


async def handle_help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await context.bot.send_message(
        chat_id=update.effective_chat.id,
        text="""
🤖 ℭ𝔥𝔞𝔱𝔊𝔓𝔗 𝔅𝔬𝔱 🤖
        
`<prompt>`: Use GPT-4O Mini to generate a response to a prompt (default)
`/gpt4 <prompt>`: Use GPT-4O to generate a response to a prompt
`/bot_prompt`: Show the current bot prompt
`/bot_prompt <prompt>`: Set the bot prompt
`/hard_reset`: Reset the conversation and all settings
`/reset`: Reset the conversation
`/temp <temperature>`: Set the chatgpt temperature
`/regen`: Regenerate the last response
`/dalle <prompt>`: Use DALL-E to generate an image
`/help`: Show this help message
        """,
        parse_mode=ParseMode.MARKDOWN,
    )


async def handle_last_response_command(
    update: Update, context: ContextTypes.DEFAULT_TYPE
):
    # upload the .txt file to telegram
    await context.bot.send_document(
        chat_id=update.effective_chat.id, document=open("latest_response.txt", "rb")
    )


def load_conversation_state():
    print(f"Loading conversation state from {CONVERSATION_STATE_FILE}")
    try:
        with open(CONVERSATION_STATE_FILE, "r") as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save():
    global conversation_state
    print(f"Saving conversation state to {CONVERSATION_STATE_FILE}")
    with open(CONVERSATION_STATE_FILE, "w") as file:
        json.dump(conversation_state, file, indent=4)


def log_events(update: Update, completion_info: dict) -> None:
    """Log the events received from Telegram."""
    print(f"logging events with completion_info: {completion_info}")
    user_info = update.message.from_user
    model = completion_info["model"]
    total_tokens = completion_info["usage"]["total_tokens"]
    prompt = completion_info["prompt"]
    with open("log.txt", "a") as f:
        line = f'{update.message.date},{user_info.username},{user_info.id},{model},{total_tokens},"{r"{}".format(prompt)}"\n'
        f.write(line)


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Log the error and send a telegram message to notify the developer."""
    # Log the error before we do anything else, so we can see it even if something breaks.
    logger.error("Exception while handling an update:", exc_info=context.error)

    # traceback.format_exception returns the usual python message about an exception, but as a
    # list of strings rather than a single string, so we have to join them together.
    tb_list = traceback.format_exception(
        None, context.error, context.error.__traceback__
    )
    tb_string = "".join(tb_list)

    # Build the message with some markup and additional information about what happened.
    # You might need to add some logic to deal with messages longer than the 4096 character limit.
    update_str = update.to_dict() if isinstance(update, Update) else str(update)
    message = (
        "An exception was raised while handling an update\n"
        f"<pre>update = {html.escape(json.dumps(update_str, indent=2, ensure_ascii=False))}"
        "</pre>\n\n"
        f"<pre>context.chat_data = {html.escape(str(context.chat_data))}</pre>\n\n"
        f"<pre>context.user_data = {html.escape(str(context.user_data))}</pre>\n\n"
        f"<pre>{html.escape(tb_string)}</pre>"
    )

    # Finally, send the message
    await context.bot.send_message(
        chat_id=update.effective_chat.id, text=message, parse_mode=ParseMode.HTML
    )


def initialize_telegram_bot():
    application = ApplicationBuilder().token(telegram_bot_key).build()
    message_handler = MessageHandler(filters.TEXT & (~filters.COMMAND), handle_message)
    image_handler = MessageHandler(filters.PHOTO, handle_image)
    gpt4_command_handler = CommandHandler("gpt4", handle_gpt4_command)
    mistral_small_command_handler = CommandHandler("ms", handle_mistral_small_command)
    mistral_medium_command_handler = CommandHandler("mm", handle_mistral_medium_command)
    bot_prompt_command_handler = CommandHandler("bot_prompt", handle_bot_prompt_command)
    reset_command_handler = CommandHandler("reset", handle_reset_command)
    hard_reset_command_handler = CommandHandler("hard_reset", handle_hard_reset_command)
    temp_command_handler = CommandHandler("temp", handle_temp_command)
    regen_command_handler = CommandHandler("regen", handle_regen_command)
    dalle_command_handler = CommandHandler("dalle", handle_dalle_command)
    help_command_handler = CommandHandler("help", handle_help_command)
    last_response_command_handler = CommandHandler(
        "last_response", handle_last_response_command
    )
    unknown_command_handler = MessageHandler(filters.COMMAND, handle_unknown_command)
    application.add_handler(message_handler)
    application.add_handler(image_handler)
    application.add_handler(gpt4_command_handler)
    application.add_handler(mistral_small_command_handler)
    application.add_handler(mistral_medium_command_handler)
    application.add_handler(bot_prompt_command_handler)
    application.add_handler(reset_command_handler)
    application.add_handler(hard_reset_command_handler)
    application.add_handler(temp_command_handler)
    application.add_handler(regen_command_handler)
    application.add_handler(dalle_command_handler)
    application.add_handler(help_command_handler)
    application.add_handler(last_response_command_handler)
    application.add_handler(unknown_command_handler)  # This must be the last handler
    application.add_error_handler(error_handler)
    application.run_polling()


def initialize_chatgpt():
    global conversation_state

    conversation_state = load_conversation_state()

    # What if the conversation state is empty?
    if conversation_state == {}:
        conversation_state = {}


if __name__ == "__main__":
    initialize_chatgpt()
    initialize_telegram_bot()
