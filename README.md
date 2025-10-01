# bg3sim

## TODO
- tags: Story/Dialog/DialogHidden, Code, unassigned, SpellCondition 등등이 있는데, checking 은 Story/Dialog만 하도록 바꾸기

- flags: global/dialog/user만 포함

## Terminologies
### Story
1. "session": 대화의 최소 단위. npc/물체와 상호작용 시 이루어지는 dialogue
2. "scenario": session들의 집합들을 order/exclusiveness constraints에 맞게 시뮬레이션한 결과
3. "chapter": scenario들의 집합들을 order/exclusiveness constraints에 맞게 시뮬레이션한 결과
4. "act": chapter들의 집합들을 order constraints에 맞게 시뮬레이션한 결과

- `output/` folder를 보면 `output/Act1`, `output/Act2b` 같이 Act라 나온게 "act"임. (Camp, Companions 등은 지금 당장은 논외. 저런 것들은 Scenario 사이사이에 들어갈 것들이긴 함. 지금 생각하기에는 너무 머리 아파서 일단 논외)
- `output/Act1/Chapel`와 같이 act의 하위 폴더들이 "chapter"임. 거의 대부분이 location(현실세계로 비유하자면 서울 같은 느낌)으로 이름이 잡혀있음
- `output/Act1/Chapel`에 있는 무수한 JSON 파일들이 "session"임. 거의 대부분이 {chapter이름}_{scenario이름}_{session이름}.json으로 구성이 되어있음.
- `output/Act1/Chapel/CHA_BronzePlaque_PAD_RichSarcophagus.json`을 예로 들어보자면 CHA: Chapel, BronzePlaque: scenario 이름 (bronze plaque 같은 사물이나 outside같은 location의 장소나 EmperorRevealed 같이 상황 설명이 이름임)
- 각 session들이 속하는 scenario별로 묶은 파일들은 `output_merged/` 폴더에 따로 만듬 
### Data
`output_merged/` 폴더가 `output/` 폴더를 아우르고 있어서 `output_merged/`에 있는 파일들을 설명할 것임

`{act이름}/{chapter이름}`안에 scenario를 시뮬레이션하기 위한 input JSON 파일들이 있음
1. "metadata": 각 session들의 메타 정보("synopsis"/"how_to_trigger")와 각각의 `output/`폴더에 있는 JSON파일 명("source_files")가 있음. 추가로 order/exclusiveness constraint을 설정한 "automatic_ordering" field가 있음. **이것을 labeling해야함**
2. 각 session마다 node들이 있음. node의 정보로는:
- "id": "{session명}_{node ID}"
- "speaker": 발화자, "text": "발화자의 utterance"
- "context": "text의 부가 설명. Cinemtic contents의 경우 이것에 대한 text description".
- "checkflags": [node에 접근하기 위해서 필요한 flags의 list]
- "setflags": [node에 접근하면 새로 추가되는 flags들의 list]
- "ruletags": [기타 rule들의 모음. 무시해도 좋음]
- "approval": [호감도가 변하는 동료들의 이름과 호감도 변화의 list]
- "rolls": "주사위 게임이라 rolls에 대한 정보가 있는데 무시해도 좋음"
- "goto, link": "JUMP to node x가 있는 경우가 있는데, 이때 가야되는 node의 id"
- "is_end": <leaf node냐 아니냐>
- "node_type": "무시해도 좋음. 궁금하면 이거에 대한 uninteresting한 정보를 알려줄 수는 있음. 문의는 DM 부탁드립니다 (기도 이모지)"
- "children": {graph traversal시 가능한 댜음 node의 종류}


## How to run
- `python act_simulator.py`: Act 전체를 시뮬레이션 하는 것. Tutorial -> Act1 -> Act1b -> Act2의 순서대로 해야하며, 현재는 Act2까지만 시뮬레이션 가능함. 왜 그런지는 DM 문의 부탁드립니다. 시작할 때 입력하는 initial flags는 `initial_flags.json`을 사용하시면 됩니다.
- `python chapter_simulator.py`: Chapter simulator
- `python scenario_simulator.py`: Scenario simulator
- `python dialog_simulator.py`: Session simulator

## Dependencies
알아서 필요한거 에러 메시지 보고 pip install하시길 바랍니다.