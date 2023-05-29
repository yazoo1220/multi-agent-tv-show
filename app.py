import streamlit as st

from collections import OrderedDict
import functools
import random
import re
import tenacity
from typing import List, Dict, Callable, Any

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate
)
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import RegexParser
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    BaseMessage,
)
from langchain.callbacks.base import BaseCallbackHandler


class DialogueAgent:
    def __init__(
        self,
        name: str,
        system_message: SystemMessage,
        model: ChatOpenAI,
    ) -> None:
        self.name = name
        self.system_message = system_message
        self.model = model
        self.prefix = f"{self.name}: "
        self.reset()
        
    def reset(self):
        self.message_history = ["Here is the conversation so far."]

    def send(self) -> str:
        """
        Applies the chatmodel to the message history
        and returns the message string
        """
        message = self.model(
            [
                self.system_message,
                HumanMessage(content="\n".join(self.message_history + [self.prefix])),
            ]
        )
        return message.content

    def receive(self, name: str, message: str) -> None:
        """
        Concatenates {message} spoken by {name} into message history
        """
        self.message_history.append(f"{name}: {message}")


class DialogueSimulator:
    def __init__(
        self,
        agents: List[DialogueAgent],
        selection_function: Callable[[int, List[DialogueAgent]], int],
    ) -> None:
        self.agents = agents
        self._step = 0
        self.select_next_speaker = selection_function
        
    def reset(self):
        for agent in self.agents:
            agent.reset()

    def inject(self, name: str, message: str):
        """
        Initiates the conversation with a {message} from {name}
        """
        for agent in self.agents:
            agent.receive(name, message)

        # increment time
        self._step += 1

    def step(self) -> tuple[str, str]:
        # 1. choose the next speaker
        speaker_idx = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_idx]

        # 2. next speaker sends message
        message = speaker.send()

        # 3. everyone receives message
        for receiver in self.agents:
            receiver.receive(speaker.name, message)

        # 4. increment time
        self._step += 1

        return speaker.name, message
        
        

class IntegerOutputParser(RegexParser):
    def get_format_instructions(self) -> str:
        return 'Your response should be an integer delimited by angled brackets, like this: <int>.'  

class DirectorDialogueAgent(DialogueAgent):
    def __init__(
        self,
        name,
        system_message: SystemMessage,
        model: ChatOpenAI,
        speakers: List[DialogueAgent],
        stopping_probability: float,
    ) -> None:
        super().__init__(name, system_message, model)
        self.speakers = speakers
        self.next_speaker = ''
        
        self.stop = False
        self.stopping_probability = stopping_probability
        self.termination_clause = 'Finish the conversation by stating a concluding message and thanking everyone.'
        self.continuation_clause = 'Do not end the conversation. Keep the conversation going by adding your own ideas.'
        
        # 1. have a prompt for generating a response to the previous speaker
        self.response_prompt_template = PromptTemplate(
            input_variables=["message_history", "termination_clause"],
            template=f"""{{message_history}}

Follow up with an insightful comment.
{{termination_clause}}
{self.prefix}
        """)
            
        # 2. have a prompt for deciding who to speak next
        self.choice_parser = IntegerOutputParser(
            regex=r'<(\d+)>',
            output_keys=['choice'],
            default_output_key='choice')        
        self.choose_next_speaker_prompt_template = PromptTemplate(
            input_variables=["message_history", "speaker_names"],
            template=f"""{{message_history}}

Given the above conversation, select the next speaker by choosing index next to their name: 
{{speaker_names}}

{self.choice_parser.get_format_instructions()}

Do nothing else.
        """)
        
        # 3. have a prompt for prompting the next speaker to speak
        self.prompt_next_speaker_prompt_template = PromptTemplate(
            input_variables=["message_history", "next_speaker"],
            template=f"""{{message_history}}

The next speaker is {{next_speaker}}. 
Prompt the next speaker to speak with an insightful question.
{self.prefix}
        """)
        
    def _generate_response(self):
        # if self.stop = True, then we will inject the prompt with a termination clause
        sample = random.uniform(0,1)
        self.stop = sample < self.stopping_probability
        
        print(f'\tStop? {self.stop}\n')
        
        response_prompt = self.response_prompt_template.format(
            message_history='\n'.join(self.message_history),
            termination_clause=self.termination_clause if self.stop else ''
        )
        
        self.response = self.model(
            [
                self.system_message,
                HumanMessage(content=response_prompt),
            ]
        ).content
                
        return self.response
        
        
    @tenacity.retry(stop=tenacity.stop_after_attempt(2),
                    wait=tenacity.wait_none(),  # No waiting time between retries
                    retry=tenacity.retry_if_exception_type(ValueError),
                    before_sleep=lambda retry_state: print(f"ValueError occurred: {retry_state.outcome.exception()}, retrying..."),
                    retry_error_callback=lambda retry_state: 0) # Default value when all retries are exhausted
    def _choose_next_speaker(self) -> str:        
        speaker_names = '\n'.join([f'{idx}: {name}' for idx, name in enumerate(self.speakers)])
        choice_prompt = self.choose_next_speaker_prompt_template.format(
            message_history='\n'.join(self.message_history + [self.prefix] + [self.response]),
            speaker_names=speaker_names
        )

        choice_string = self.model(
            [
                self.system_message,
                HumanMessage(content=choice_prompt),
            ]
        ).content
        choice = int(self.choice_parser.parse(choice_string)['choice'])
        
        return choice
    
    def select_next_speaker(self):
        return self.chosen_speaker_id
            
    def send(self) -> str:
        """
        Applies the chatmodel to the message history
        and returns the message string
        """
        # 1. generate and save response to the previous speaker
        self.response = self._generate_response()
        
        if self.stop:
            message = self.response 
        else:
            # 2. decide who to speak next
            self.chosen_speaker_id = self._choose_next_speaker()
            self.next_speaker = self.speakers[self.chosen_speaker_id]
            print(f'\tNext speaker: {self.next_speaker}\n')

            # 3. prompt the next speaker to speak
            next_prompt = self.prompt_next_speaker_prompt_template.format(
                message_history="\n".join(self.message_history + [self.prefix] + [self.response]),
                next_speaker=self.next_speaker
            )
            message = self.model(
                [
                    self.system_message,
                    HumanMessage(content=next_prompt),
                ]
            ).content
            message = ' '.join([self.response, message])
            
        return message
        
        

st.title('さんま御殿メーカー')
topic = st.text_input('topic','踊るさんま御殿　有名人夫を転がす奥様スペシャル:太田光（爆笑問題）の妻・太田光代と、田中裕二（爆笑問題）の妻・山口もえがテレビ初共演。山口は「社長のおかげで我が家はなりたってます」と太田に感謝。しかし二人の夫に対する不満が爆発！')
st.markdown('---')
director_name = st.text_input('司会者',"さんま")
director_role = st.text_input('役割','踊るさんま御殿の司会者')
director_character = st.text_input('キャラ','どんなネタも面白くしてしまう最強のコメディアン。関西弁でノリツッコミが得意。')
st.markdown('---')
agent1_name = st.text_input('ゲスト1',"太田光代")
agent1_role = st.text_input('役割','爆笑問題の太田光の嫁')
agent1_character = st.text_input('キャラ','切れ者で歯に衣着せぬ言い方が評判')
st.markdown('---')
agent2_name = st.text_input('ゲスト2',"山口もえ")
agent2_role = st.text_input('役割','爆笑問題の田中裕二の嫁')
agent2_character = st.text_input('キャラ','おっとりしているが言うことは言う')
st.markdown('---')
agent3_name = st.text_input('ゲスト3',"高木泰弘")
agent3_role = st.text_input('役割','観客')
agent3_character = st.text_input('キャラ','リアクション芸人')
st.markdown('---')

agent_summaries = OrderedDict({
    director_name: (director_role, director_character),
    agent1_name: (agent1_role, agent1_character),
    agent2_name: (agent2_role, agent2_character),
    agent3_name: (agent3_role, agent3_character),
})
word_limit = 50

start_button = st.button('はじめる')

if start_button:
    with st.spinner('準備中...'):
      agent_summary_string = '\n- '.join([''] + [f'{name}: {role}, who is {location}' for name, (role, location) in agent_summaries.items()])

      conversation_description = f"""This is a Daily Show episode discussing the following topic: {topic}.

      The episode features {agent_summary_string}."""

      agent_descriptor_system_message = SystemMessage(
          content="You can add detail to the description of each person.")

      def generate_agent_description(agent_name, agent_role, agent_location):
          agent_specifier_prompt = [
              agent_descriptor_system_message,
              HumanMessage(content=
                  f"""{conversation_description}
                  Please reply with a creative description of {agent_name}, who is a {agent_role} in {agent_location}, that emphasizes their particular role and location.
                  Speak directly to {agent_name} in {word_limit} words or less.
                  Do not add anything else."""
                  )
          ]
          agent_description = ChatOpenAI(temperature=1.0)(agent_specifier_prompt).content
          return agent_description

      def generate_agent_header(agent_name, agent_role, agent_location, agent_description):
          return f"""{conversation_description}

      Your name is {agent_name}, your role is {agent_role}, and you are located in {agent_location}.

      Your description is as follows: {agent_description}

      You are discussing the topic: {topic}.

      Your goal is to provide the funniest, creative, and silly perspectives of the topic from the perspective of your role and your character. Please speak in Japanese
      """

      def generate_agent_system_message(agent_name, agent_header):
          return SystemMessage(content=(
          f"""{agent_header}
      You will speak in the style of {agent_name}, and exaggerate your personality.
      Do not say the same things over and over again.
      Speak in the first person from the perspective of {agent_name}
      For describing your own body movements, wrap your description in '*'.
      Do not change roles!
      Do not speak from the perspective of anyone else.
      Speak only from the perspective of {agent_name}.
      Stop speaking the moment you finish speaking from your perspective.
      Never forget to keep your response to {word_limit} words!
      Do not add anything else. Please speak in Japanese
          """
          ))

      agent_descriptions = [generate_agent_description(name, role, location) for name, (role, location) in agent_summaries.items()]
      agent_headers = [generate_agent_header(name, role, location, description) for (name, (role, location)), description in zip(agent_summaries.items(), agent_descriptions)]
      agent_system_messages = [generate_agent_system_message(name, header) for name, header in zip(agent_summaries, agent_headers)]


      for name, description, header, system_message in zip(agent_summaries, agent_descriptions, agent_headers, agent_system_messages):
          with st.expander('ℹ️'):
              st.write(f'\n\n{name} Description:')
              st.write(f'\n{description}')
              st.write(f'\nHeader:\n{header}')
              st.write(f'\nSystem Message:\n{system_message.content}')



      topic_specifier_prompt = [
          SystemMessage(content="You can make a task more specific."),
          HumanMessage(content=
              f"""{conversation_description}

              Please elaborate on the topic. 
              Frame the topic as a single question to be answered.
              Be silly, creative and imaginative.
              Please reply with the specified topic in {word_limit} words or less. 
              Do not add anything else. this should be in Japanese"""
              )
      ]
      specified_topic = ChatOpenAI(temperature=1.0)(topic_specifier_prompt).content

      print(f"Original topic:\n{topic}\n")


      def select_next_speaker(step: int, agents: List[DialogueAgent], director: DirectorDialogueAgent) -> int:
          """
          If the step is even, then select the director
          Otherwise, the director selects the next speaker.
          """    
          # the director speaks on odd steps
          if step % 2 == 1:
              idx = 0
          else:
              # here the director chooses the next speaker
              idx = director.select_next_speaker() + 1  # +1 because we excluded the director
          return idx

      print(f"Detailed topic:\n{specified_topic}\n")


      class SimpleStreamlitCallbackHandler(BaseCallbackHandler):
          """ Copied only streaming part from StreamlitCallbackHandler """

          def __init__(self) -> None:
              self.tokens_area = st.empty()
              self.tokens_stream = ""

          def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
              """Run on new LLM token. Only available when streaming is enabled."""
              self.tokens_stream += token
              self.tokens_area.markdown(self.tokens_stream)

      handler = SimpleStreamlitCallbackHandler()



      director = DirectorDialogueAgent(
          name=director_name,
          system_message=agent_system_messages[0],
          model=ChatOpenAI(temperature=0.5,model_name='gpt-3.5-turbo', callbacks=[handler]),
          speakers=[name for name in agent_summaries if name != director_name],
          stopping_probability=0.2
      )

      agents = [director]
      for name, system_message in zip(list(agent_summaries.keys())[1:], agent_system_messages[1:]):        
          agents.append(DialogueAgent(
              name=name,
              system_message=system_message,
              model=ChatOpenAI(temperature=0.5,model_name='gpt-3.5-turbo',callbacks=[handler]),
          ))

      simulator = DialogueSimulator(
          agents=agents,
          selection_function=functools.partial(select_next_speaker, director=director)
      )
      simulator.reset()
      simulator.inject('Audience member', specified_topic)
      print(f"(Audience member): {specified_topic}")
      print('\n')

      area = st.empty()
      while True:
          name, message = simulator.step()
          area.markdown(f"({name}): {message}")
          area.markdown('\n')
          if director.stop:
              break
