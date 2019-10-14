#! -*- coding: utf-8 -*-
import random

episode = [
    ['sleep', 'bag', 'face', 'weather', 'game'],
    ['study', 'sleep', 'game', 'chat', 'time'],
    ['intelligent_home', 'food', 'greet', 'chat', 'weather'],
    ['query', 'food', 'sleep', 'volume', 'chat'],
    ['command', 'machine', 'radio', 'study', 'sleep'],
    ['command', 'volume', 'study', 'chat', 'query'],
    ['game', 'intelligent_home', 'story', 'machine', 'weather'],
    ['food', 'game', 'query', 'command', 'sleep'],
    ['intelligent_home', 'bag', 'face', 'study', 'sleep'],
    ['sleep', 'food', 'time', 'news', 'chat'],
    ['study', 'sleep', 'game', 'command', 'radio'],
    ['command', 'game', 'music', 'alarm', 'machine'],
    ['sleep', 'game', 'volume', 'story', 'bag'],
    ['music', 'command', 'study', 'alarm', 'time'],
    ['machine', 'story', 'food', 'weather', 'query'],
    ['alarm', 'machine', 'game', 'chat', 'command'],
    ['radio', 'weather', 'time', 'greet', 'sleep'],
    ['intelligent_home', 'weather', 'news', 'greet', 'music'],
    ['music', 'time', 'command', 'radio', 'study'],
    ['command', 'chat', 'volume', 'game', 'food']
]

labels = ['alarm', 'bag', 'chat', 'command', 'face', 'food', 'game', 'greet', 'intelligent_home', 'machine', 'music',
          'news', 'query', 'radio', 'sleep', 'story', 'study', 'time', 'volume', 'weather']

for i in range(20):
    print(random.sample(labels, k=5))


