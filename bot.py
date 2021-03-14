import nltk
import bot_config
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
import random
import json
import copy22
import aiogram
import logging

from aiogram import Bot, Dispatcher, executor, types

BOT_CONFIG = bot_config.BOT_CONFIG
dialogues_structured_cut = copy22.dialogues
 

#задаём уровень логов
logging.basicConfig(level=logging.INFO)

#инициализируем бота
bot = Bot(token=bot_config.TOKEN)
dp = Dispatcher(bot)

def clear_phrase(phrase):
    phrase = phrase.lower()
    alphabet = 'йцукенгшщзхъфывапролджэячсмитьбюё- '
    result = ''.join(symbol for symbol in phrase if symbol in alphabet)
    # for symbol in phrase:
    #     if symbol in alphabet:
    #         result += symbol
    return result.strip()

### set X_text and y ###

X_text = []
y = []

for intent, intent_data in BOT_CONFIG['intents'].items():
    for example in intent_data['examples']:
        X_text.append(example)
        y.append(intent)


### Веторизация ###

vectorizer = TfidfVectorizer(analyzer='char', ngram_range=(3, 3))
X = vectorizer.fit_transform(X_text)
vectorizer.get_feature_names()

### Классификация ###

clf = LinearSVC(random_state=0)
clf.fit(X, y)

def get_failure_phrase():
    failure_phrases = BOT_CONFIG['failure_phrases']
    return random.choice(failure_phrases)

def classify_intent(replica):
    replica = clear_phrase(replica)
    intent = clf.predict(vectorizer.transform([replica]))[0]

    for example in BOT_CONFIG['intents'][intent]['examples']:
        example = clear_phrase(example)
        distance = nltk.edit_distance(replica, example)
        if example and distance / len(example) <= 0.5:
            return intent

def get_answer_by_intent(intent):
    if intent in BOT_CONFIG['intents']:
        responses = BOT_CONFIG['intents'][intent]['responses'] 
        return random.choice(responses)

# with open('dialogues.txt') as f:
#     content = f.read()

# dialogues_str = content.split('\n\n')
# dialogues = [dialogue_str.split('\n')[:2] for dialogue_str in dialogues_str]

# ### Фильтр dialogues ###
 
# dialogues_filtered = []
# questions = set()
# for dialogue in dialogues:
#     if len(dialogue) != 2:
#         continue
#     question, answer = dialogue
#     question = clear_phrase(question[2:])
#     answer = answer[2:]
#     if question != '' and question not in questions:
#         questions.add(question)
#         dialogues_filtered.append([question, answer])

# ### Structured dialogues ###
# dialogues_structured = {}

# for question, answer in dialogues_filtered:
#     words = set(question.split(' '))
#     for word in words:
#         if word not in dialogues_structured:
#             dialogues_structured[word] = []
#         dialogues_structured[word].append([question, answer]) 

# ### Cutting structured dialogues ###
# dialogues_structured_cut = {}

# for word, pairs in dialogues_structured.items():
#     pairs.sort(key=lambda pair: len(pair[0]))
#     dialogues_structured_cut[word] = pairs[:1000]

# with open('data.json', 'w') as fp:
#     json.dump(dialogues_structured_cut, fp)


# dialogues_structured_cut = {}
# with open('data.json') as p:
#     dialogues_structured_cut = json.load(p)
# p.close()

# with open("copy.py", "w") as text_file:
#     text_file.write('dialogues' + ' = ' + str(dialogues_structured_cut))

def generate_answer(replica):
    replica = clear_phrase(replica)
    words = set(replica.split(' '))
    mini_dataset = []

    for word in words:
        if word in dialogues_structured_cut:
            mini_dataset += dialogues_structured_cut[word]
    
    #TODO убрать повторы из датасета 
 
    answers = [] # [[distanse_wighted, question, answer]]

    for question, answer in mini_dataset:
        if abs(len(replica) - len(question)) / len(question) < 0.2:
            distance = nltk.edit_distance(replica, question)
            distance_weight = distance / len(question)
            if distance_weight < 0.2:
                answers.append([distance_weight, question, answer])

    if answers:
        answer = min(answers, key=lambda three: three[0])[2]
        return answer


def classify_intent(replica):
    replica = clear_phrase(replica)
    for intent, intent_data in BOT_CONFIG['intents'].items():
        for example in intent_data['examples']:
            example = clear_phrase(example)
            distance = nltk.edit_distance(replica, example)
            if example and distance / len(example) < 0.4:
                return intent 


stats = {'intent': 0, 'generate': 0, 'failure': 0}

def bot(replica):
    #NLU
    intent = classify_intent(replica)

    #Answer generation

    # выбор заготовленной реплики
    if intent:
        answer = get_answer_by_intent(intent)
        if answer:
            stats['intent'] += 1
            return answer
    
    # вызов генеративной модели
    answer = generate_answer(replica)
    if answer:
        stats['generate'] += 1
        return answer
    
    # берём заглушку 
    stats['failure'] += 1
    return get_failure_phrase()



@dp.message_handler(commands=['start'])
async def echo(message: types.Message):
    username = message.from_user.username
    await message.answer('Привет, ' + username + '. Я умный бот. Давай поговрим')

@dp.message_handler(content_types=['text'])
async def lalala(message: types.Message):
    await message.answer(bot(message.text))
    print(stats)

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
    
