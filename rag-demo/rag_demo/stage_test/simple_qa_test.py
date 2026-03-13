from ai_starter.langchain import LangChainChatZhipuAI
from ai_starter import get_logger
from langchain_core.messages import (
    AIMessage,  # 等价于OpenAI接口中的assistant role
    HumanMessage,  # 等价于OpenAI接口中的user role
    SystemMessage  # 等价于OpenAI接口中的system role
)

from ..model import Date

from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder
)

logger = get_logger(__name__)


def single_turn_test(chat_model: LangChainChatZhipuAI):
    response = chat_model.invoke("你是谁?")
    logger.info(response.content)

def multi_turn_test(chat_model: LangChainChatZhipuAI):
    messages = [
        SystemMessage(content="你是京东人工客服 东子"),
        HumanMessage(content="我是客户步惊云"),
        AIMessage(content="欢迎！ 我是京东客服东子"),
        HumanMessage(content="你是谁？")
    ]

    # 直接输出
    # response = chat_model.invoke(messages)
    # logger.info(response.content)

    # 流式输出
    for token in chat_model.stream("你是谁"):
        logger.info(token.content)

def single_turn_prompt_test(chat_model: LangChainChatZhipuAI):
    template = PromptTemplate.from_template("给我讲个关于{subject}的笑话")
    prompt = template.format(subject='小明')
    logger.info(prompt)
    response = chat_model.invoke(prompt)
    logger.info(response.content)

def multi_turn_prompt_test(chat_model: LangChainChatZhipuAI):
    template = ChatPromptTemplate.from_messages(
        [
            SystemMessagePromptTemplate.from_template("你是{company}客服公司. 你的名字叫{name}"),
            HumanMessagePromptTemplate.from_template("{query}")
        ]
    )

    prompt = template.format_messages(company="京东", name="东子", query="你是谁")
    logger.info(prompt)
    response = chat_model.invoke(prompt)
    logger.info(response.content)

def multi_turn_messages_place_holder_test(chat_model: LangChainChatZhipuAI):

    human_template = "Translate your answer to {language}"
    human_message_template = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [MessagesPlaceholder("chat_messages_history"), human_message_template]
    )

    human_message = HumanMessage(content="Who is Elon Musk")
    ai_message = AIMessage(content="Elon Musk is a billionaire entrepreneur, inventor, and industrial designer")

    messages_prompt = chat_prompt.format_prompt(
        chat_messages_history=[human_message, ai_message],
        language="中文"
    )

    logger.info(messages_prompt.to_messages())
    response = chat_model.invoke(messages_prompt)
    logger.info(response.content)

def prompt_from_file_test(chat_model: LangChainChatZhipuAI):
    file_template = PromptTemplate.from_file("../../prompt/example_prompt.txt", encoding="utf-8")
    prompt = file_template.format(topic="黑色幽默")
    logger.info(prompt)
    response = chat_model.invoke(prompt)

    # response是AIMessage对象
    logger.info(response.content)

def class_format_output(chat_model: LangChainChatZhipuAI):
    structured_chat_model = chat_model.with_structured_output(Date)
    template =  """提取用户输入的日期.
    用户输入:{input}
    """

    prompt = PromptTemplate(
        template=template
    )

    input = "2023年四月6日天气晴"
    input_prompt = prompt.format(input=input)

    # response是自定义的Date对象
    response = structured_chat_model.invoke(input_prompt)
    logger.info(f"提取的日期: year={response.year}, month={response.month}, day={response.day}")


def main():
    chat_model = LangChainChatZhipuAI()
    # single_turn_test(chat_model)
    # multi_turn_test(chat_model)
    # single_turn_prompt_test(chat_model)
    # multi_turn_prompt_test(chat_model)
    # multi_turn_messages_place_holder_test(chat_model)
    # prompt_from_file_test(chat_model)
    class_format_output(chat_model)


if __name__ == "__main__":
    main()