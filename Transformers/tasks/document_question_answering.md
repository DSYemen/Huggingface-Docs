## الإجابة على الأسئلة الواردة في المستند

[[open-in-colab]]

الإجابة على الأسئلة الواردة في المستند، والتي يشار إليها أيضًا باسم الإجابة المرئية على الأسئلة الواردة في المستند، هي مهمة تتضمن تقديم إجابات على الأسئلة المطروحة حول صور المستندات. المدخلات إلى النماذج التي تدعم هذه المهمة هي عادة مزيج من الصورة والسؤال، والناتج هو إجابة معبر عنها باللغة الطبيعية. تستخدم هذه النماذج طرائق متعددة، بما في ذلك النص، ومواضع الكلمات (حدود الإحداثيات)، والصورة نفسها.

يوضح هذا الدليل كيفية:

- ضبط نموذج [LayoutLMv2](../model_doc/layoutlmv2) الدقيق على مجموعة بيانات [DocVQA](https://huggingface.co/datasets/nielsr/docvqa_1200_examples_donut).
- استخدام النموذج المضبوط دقيقًا للاستنتاج.

<Tip>

لرؤية جميع التصميمات ونقاط التحقق المتوافقة مع هذه المهمة، نوصي بالتحقق من [صفحة المهمة](https://huggingface.co/tasks/image-to-text).

</Tip>

يحل LayoutLMv2 مشكلة الإجابة على الأسئلة في المستند عن طريق إضافة رأس للإجابة على الأسئلة أعلى حالات الرموز النهائية، للتنبؤ بمواضع الرموز الأولية والنهائية للإجابة. وبعبارة أخرى، تتم معاملة المشكلة على أنها إجابة استخلاصية على الأسئلة: استخراج قطعة المعلومات التي تجيب على السؤال، بالنظر إلى السياق. يأتي السياق من إخراج محرك التعرف البصري على الحروف (OCR)، وهنا هو برنامج Tesseract من Google.

قبل البدء، تأكد من تثبيت جميع المكتبات الضرورية. يعتمد LayoutLMv2 على detectron2 وtorchvision وtesseract.

```bash
pip install -q transformers datasets
```

```bash
pip install 'git+https://github.com/facebookresearch/detectron2.git'
pip install torchvision
```

```bash
sudo apt install tesseract-ocr
pip install -q pytesseract
```

بمجرد تثبيت جميع التبعيات، أعد تشغيل وقت التشغيل الخاص بك.

نحن نشجعك على مشاركة نموذجك مع المجتمع. قم بتسجيل الدخول إلى حساب Hugging Face الخاص بك لتحميله إلى 🤗 Hub.

عند مطالبتك، أدخل رمزك للتسجيل:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

دعونا نحدد بعض المتغيرات العالمية.

```py
>>> model_checkpoint = "microsoft/layoutlmv2-base-uncased"
>>> batch_size = 4
```

## تحميل البيانات

في هذا الدليل، نستخدم عينة صغيرة من DocVQA المعالجة مسبقًا والتي يمكنك العثور عليها على 🤗 Hub. إذا كنت ترغب في استخدام مجموعة DocVQA الكاملة، فيمكنك التسجيل وتنزيلها من [الصفحة الرئيسية لـ DocVQA](https://rrc.cvc.uab.es/?ch=17). إذا قمت بذلك، لمتابعة هذا الدليل، راجع [كيفية تحميل الملفات إلى مجموعة بيانات 🤗](https://huggingface.co/docs/datasets/loading#local-and-remote-files).

```py
>>> from datasets import load_dataset

>>> dataset = load_dataset("nielsr/docvqa_1200_examples")
>>> dataset
DatasetDict({
train: Dataset({
features: ['id', 'image', 'query', 'answers', 'words', 'bounding_boxes', 'answer'],
num_rows: 1000
})
test: Dataset({
features: ['id', 'image', 'query', 'answers', 'words', 'bounding_boxes', 'answer'],
num_rows: 200
})
})
```

كما ترى، تم تقسيم مجموعة البيانات بالفعل إلى مجموعات تدريب واختبار. الق نظرة على مثال عشوائي للتعرف على الميزات.

```py
>>> dataset["train"].features
```

هذا ما تمثله الحقول الفردية:

- `id`: معرف المثال
- `image`: كائن PIL.Image.Image يحتوي على صورة المستند
- `query`: سلسلة الاستعلام - سؤال اللغة الطبيعية المطروحة، بعدة لغات
- `answers`: قائمة الإجابات الصحيحة التي قدمها معلّقو البيانات
- `words` و`bounding_boxes`: نتائج التعرف البصري على الحروف، والتي لن نستخدمها هنا
- `answer`: إجابة تمت مطابقتها بواسطة نموذج مختلف لن نستخدمه هنا

دعونا نترك فقط الأسئلة باللغة الإنجليزية، ونقوم بإسقاط ميزة "الإجابة" التي يبدو أنها تحتوي على تنبؤات بنموذج آخر. سنأخذ أيضًا الإجابة الأولى من مجموعة الإجابات المقدمة من المعلمين. أو يمكنك أخذ عينة عشوائية منها.

```py
>>> updated_dataset = dataset.map(lambda example: {"question": example["query"]["en"]}, remove_columns=["query"])
>>> updated_dataset = updated_dataset.map(
...     lambda example: {"answer": example["answers"][0]}, remove_columns=["answer", "answers"]
... )
```

لاحظ أن نقطة التحقق LayoutLMv2 التي نستخدمها في هذا الدليل تم تدريبها باستخدام `max_position_embeddings = 512` (يمكنك العثور على هذه المعلومات في ملف `config.json` الخاص بنقطة التحقق [هنا](https://huggingface.co/microsoft/layoutlmv2-base-uncased/blob/main/config.json#L18)).

يمكننا تقليص الأمثلة ولكن لتجنب الموقف الذي قد تكون فيه الإجابة في نهاية مستند طويل وتنتهي بالاقتطاع، هنا سنزيل الأمثلة القليلة التي من المحتمل أن ينتهي فيها تضمينها إلى أكثر من 512.

إذا كانت معظم المستندات في مجموعة البيانات الخاصة بك طويلة، فيمكنك تنفيذ إستراتيجية النافذة المنزلقة - راجع [هذا الدفتر](https://github.com/huggingface/notebooks/blob/main/examples/question_answering.ipynb) للحصول على التفاصيل.

```py
>>> updated_dataset = updated_dataset.filter(lambda x: len(x["words"]) + len(x["question"].split()) < 512)
```

في هذه المرحلة، دعنا نزيل أيضًا ميزات التعرف البصري على الحروف من هذه المجموعة من البيانات. هذه هي نتيجة التعرف البصري على الحروف لضبط نموذج مختلف. لا يزالون بحاجة إلى بعض المعالجة إذا أردنا استخدامها، حيث لا تتطابق مع متطلبات الإدخال للنموذج الذي نستخدمه في هذا الدليل. بدلاً من ذلك، يمكننا استخدام [`LayoutLMv2Processor`] على البيانات الأصلية لكل من التعرف البصري على الحروف والتوكنيز. بهذه الطريقة، سنحصل على الإدخالات التي تتطابق مع الإدخال المتوقع للنموذج. إذا كنت تريد معالجة الصور يدويًا، فراجع وثائق نموذج [`LayoutLMv2`](../model_doc/layoutlmv2) لمعرفة تنسيق الإدخال الذي يتوقعه النموذج.

```py
>>> updated_dataset = updated_dataset.remove_columns("words")
>>> updated_dataset = updated_dataset.remove_columns("bounding_boxes")
```

أخيرًا، لن تكتمل استكشاف البيانات إذا لم نلقي نظرة على مثال صورة.

```py
>>> updated_dataset["train"][11]["image"]
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/docvqa_example.jpg" alt="مثال DocVQA"/>
</div>

## معالجة البيانات مسبقًا

مهمة الإجابة على الأسئلة الواردة في المستند هي مهمة متعددة الوسائط، ويجب التأكد من معالجة الإدخالات من كل وسيط وفقًا لتوقعات النموذج. دعونا نبدأ بتحميل [`LayoutLMv2Processor`]، والذي يجمع داخليًا بين معالج الصور الذي يمكنه التعامل مع بيانات الصور ومعالج الترميز الذي يمكنه تشفير بيانات النص.

```py
>>> from transformers import AutoProcessor

>>> processor = AutoProcessor.from_pretrained(model_checkpoint)
```

### معالجة صور المستندات

أولاً، دعونا نعد صور المستندات للنموذج بمساعدة `image_processor` من المعالج.

بشكل افتراضي، يقوم معالج الصور بإعادة تحجيم الصور إلى 224x224، والتأكد من أن لديها الترتيب الصحيح لقنوات الألوان، وتطبيق التعرف البصري على الحروف باستخدام برنامج Tesseract للحصول على الكلمات وحدود الإحداثيات المعيارية. في هذا البرنامج التعليمي، كل هذه الافتراضيات هي بالضبط ما نحتاجه.

اكتب دالة تطبق معالجة الصور الافتراضية على دفعة من الصور وتعيد نتائج التعرف البصري على الحروف.

```py
>>> image_processor = processor.image_processor


>>> def get_ocr_words_and_boxes(examples):
...     images = [image.convert("RGB") for image in examples["image"]]
...     encoded_inputs = image_processor(images)

...     examples["image"] = encoded_inputs.pixel_values
...     examples["words"] = encoded_inputs.words
...     examples["boxes"] = encoded_inputs.boxes

...     return examples
```

لتطبيق هذه المعالجة المسبقة على مجموعة البيانات بأكملها بطريقة سريعة، استخدم [`~datasets.Dataset.map`] .

```py
>>> dataset_with_ocr = updated_dataset.map(get_ocr_words_and_boxes, batched=True, batch_size=2)
```
### معالجة نص البيانات 
بعد تطبيق التعرف البصري على الحروف على الصور، نحتاج إلى تشفير الجزء النصي من مجموعة البيانات لإعدادها للنمذجة.
وينطوي ذلك على تحويل الكلمات والصناديق التي حصلنا عليها في الخطوة السابقة إلى مستوى الرمز `input_ids` و`attention_mask` و`token_type_ids` و`bbox`. ولمعالجة النص، سنحتاج إلى `tokenizer` من المعالج.
```py
>>> tokenizer = processor.tokenizer
```
بالإضافة إلى المعالجة المذكورة أعلاه، نحتاج أيضًا إلى إضافة التسميات للنمذجة. وبالنسبة لنماذج `xxxForQuestionAnswering` في 🤗 Transformers، تتكون التسميات من `start_positions` و`end_positions`، والتي تشير إلى الرمز الذي يمثل بداية الإجابة ونهايتها.
دعونا نبدأ بذلك. قم بتعريف دالة مساعدة يمكنها العثور على قائمة فرعية (الإجابة مقسمة إلى كلمات) في قائمة أكبر (قائمة الكلمات).
ستأخذ هذه الدالة قائمتين كمدخلات، `words_list` و`answer_list`. ثم ستتكرر عبر `words_list` والتحقق مما إذا كانت الكلمة الحالية في `words_list` (words_list[i]) تساوي الكلمة الأولى من answer_list (answer_list[0]) وإذا كانت القائمة الفرعية من `words_list` بدءًا من الكلمة الحالية وطولها يساوي طول `answer_list` يساوي `answer_list`.
إذا كان هذا الشرط صحيحًا، فهذا يعني أنه تم العثور على تطابق، وستقوم الدالة بتسجيل التطابق وفهرس البداية الخاص به (idx) وفهرس النهاية (idx + len(answer_list) - 1). إذا تم العثور على أكثر من تطابق واحد، فستعيد الدالة التطابق الأول فقط.
إذا لم يتم العثور على أي تطابق، تقوم الدالة بإرجاع (None، 0، و0).
```py
>>> def subfinder(words_list, answer_list):
...     matches = []
...     start_indices = []
...     end_indices = []
...     for idx, i in enumerate(range(len(words_list))):
...         if words_list[i] == answer_list[0] and words_list[i : i + len(answer_list)] == answer_list:
...             matches.append(answer_list)
...             start_indices.append(idx)
...             end_indices.append(idx + len(answer_list) - 1)
...     if matches:
...         return matches[0], start_indices[0], end_indices[0]
...     else:
...         return None, 0, 0
```
ولتوضيح كيفية عثور هذه الدالة على موضع الإجابة، دعونا نستخدمها في مثال:
```py
>>> example = dataset_with_ocr["train"][1]
>>> words = [word.lower() for word in example["words"]]
>>> match, word_idx_start, word_idx_end = subfinder(words, example["answer"].lower().split())
>>> print("Question: ", example["question"])
>>> print("Words:", words)
>>> print("Answer: ", example["answer"])
>>> print("start_index", word_idx_start)
>>> print("end_index", word_idx_end)
Question:  من هو في سي سي في هذه الرسالة؟
الكلمات: ['wie', 'baw', 'brown', '&', 'williamson', 'tobacco', 'corporation', 'research', '&', 'development', 'internal', 'correspondence', 'to:', 'r.', 'h.', 'honeycutt', 'ce:', 't.f.', 'riehl', 'from:', '.', 'c.j.', 'cook', 'date:', 'may', '8,', '1995', 'subject:', 'review', 'of', 'existing', 'brainstorming', 'ideas/483', 'the', 'major', 'function', 'of', 'the', 'product', 'innovation', 'graup', 'is', 'to', 'develop', 'marketable', 'nove!', 'products', 'that', 'would', 'be', 'profitable', 'to', 'manufacture', 'and', 'sell.', 'novel', 'is', 'defined', 'as:', 'of', 'a', 'new', 'kind,', 'or', 'different', 'from', 'anything', 'seen', 'or', 'known', 'before.', 'innovation', 'is', 'defined', 'as:', 'something', 'new', 'or', 'different', 'introduced;', 'act', 'of', 'innovating;', 'introduction', 'of', 'new', 'things', 'or', 'methods.', 'the', 'products', 'may', 'incorporate', 'the', 'latest', 'technologies,', 'materials', 'and', 'know-how', 'available', 'to', 'give', 'then', 'a', 'unique', 'taste', 'or', 'look.', 'the', 'first', 'task', 'of', 'the', 'product', 'innovation', 'group', 'was', 'to', 'assemble,', 'review', 'and', 'categorize', 'a', 'list', 'of', 'existing', 'brainstorm
ing', 'ideas.', 'ideas', 'were', 'grouped', 'into', 'two', 'major', 'categories', 'labeled', 'appearance', 'and', 'taste/aroma.', 'these', 'categories', 'are', 'used', 'for', 'novel', 'products', 'that', 'may', 'differ', 'from', 'a', 'visual', 'and/or', 'taste/aroma', 'point', 'of', 'view', 'compared', 'to', 'canventional', 'cigarettes.', 'other', 'categories', 'include', 'a', 'combination', 'of', 'the', 'above,', 'filters,', 'packaging', 'and', 'brand', 'extensions.', 'appearance', 'this', 'category', 'is', 'used', 'for', 'novel', 'cigarette', 'constructions', 'that', 'yield', 'visually', 'different', 'products', 'with', 'minimal', 'changes', 'in', 'smoke', 'chemistry', 'two', 'cigarettes', 'in', 'cne.', 'emulti-plug', 'te', 'build', 'yaur', 'awn', 'cigarette.', 'eswitchable', 'menthol', 'or', 'non', 'menthol', 'cigarette.', '*cigarettes', 'with', 'interspaced', 'perforations', 'to', 'enable', 'smoker', 'to', 'separate', 'unburned', 'section', 'for', 'future', 'smoking.', '«short', 'cigarette,', 'tobacco', 'section', '30', 'mm.', '«extremely', 'fast', 'buming', 'cigarette.', '«novel', 'cigarette', 'constructions', 'that', 'permit', 'a', 'significant', 'reduction', 'iretobacco', 'weight', 'while', 'maintaining', 'smoking', 'mechanics', 'and', 'visual', 'characteristics.', 'higher', 'basis', 'weight', 'paper:', 'potential', 'reduction', 'in', 'tobacco', 'weight.', '«more', 'rigid', 'tobacco', 'column;', 'stiffing', 'agent', 'for', 'tobacco;', 'e.g.', 'starch', '*colored', 'tow', 'and', 'cigarette', 'papers;', 'seasonal', 'promotions,', 'e.g.', 'pastel', 'colored', 'cigarettes', 'for', 'easter', 'or', 'in', 'an', 'ebony', 'and', 'ivory', 'brand', 'containing', 'a', 'mixture', 'of', 'all', 'black', '(black', 'paper', 'and', 'tow)', 'and', 'ail', 'white', 'cigarettes.']
الإجابة:  ت.ف. رييل
فهرس البداية 17
فهرس النهاية 18
```
ومع ذلك، بعد الترميز، ستبدو الأمثلة على النحو التالي:
```py
>>> encoding = tokenizer(example["question"], example["words"], example["boxes"])
>>> tokenizer.decode(encoding["input_ids"])
[CLS] من هو في سي سي في هذه الرسالة؟ [SEP] wie baw brown & williamson tobacco corporation research & development ...
```
سنحتاج إلى العثور على موضع الإجابة في الإدخال المشفر.
* `token_type_ids` تخبرنا بالرموز التي تعد جزءًا من السؤال، وأيها جزء من كلمات المستند.
* `tokenizer.cls_token_id` سيساعد في العثور على الرمز الخاص في بداية الإدخال.
* `word_ids` ستساعد في مطابقة الإجابة الموجودة في الكلمات الأصلية `words` مع نفس الإجابة في الإدخال المشفر وتحديد موضع البداية/النهاية للإجابة في الإدخال المشفر.
مع أخذ ذلك في الاعتبار، دعونا ننشئ دالة لتشفير دفعة من الأمثلة في مجموعة البيانات:
```py
>>> def encode_dataset(examples, max_length=512):
...     questions = examples["question"]
...     words = examples["words"]
...     boxes = examples["boxes"]
...     answers = examples["answer"]

...     # قم بتشفير دفعة الأمثلة وبدء تهيئة مواقع البداية والنهاية
...     encoding = tokenizer(questions, words, boxes, max_length=max_length, padding="max_length", truncation=True)
...     start_positions = []
...     end_positions = []

...     # حلق عبر الأمثلة في الدفعة
...     for i in range(len(questions)):
...         cls_index = encoding["input_ids"][i].index(tokenizer.cls_token_id)

...         # العثور على موضع الإجابة في كلمات المثال
...         words_example = [word.lower() for word in words[i]]
...         answer = answers[i]
...         match, word_idx_start, word_idx_end = subfinder(words_example, answer.lower().split())

...         if match:
...             # إذا تم العثور على تطابق، استخدم `token_type_ids` للعثور على المكان الذي تبدأ منه الكلمات في الترميز
...             token_type_ids = encoding["token_type_ids"][i]
...             token_start_index = 0
...             while token_type_ids[token_start_index] != 1:
...                 token_start_index += 1

...             token_end_index = len(encoding["input_ids"][i]) - 1
...             while token_type_ids[token_end_index] != 1:
...                 token_end_index -= 1

...             word_ids = encoding.word_ids(i)[token_start_index : token_end_index + 1]
...             start_position = cls_index
...             end_position = cls_index

...             # قم بالحلقة فوق word_ids وزيادة `token_start_index` حتى تتطابق مع موضع الإجابة في الكلمات
...             # بمجرد حدوث التطابق، احفظ `token_start_index` كـ `start_position` للإجابة في الترميز
...             for id in word_ids:
...                 if id == word_idx_start:
...                     start_position = token_start_index
...                 else:
...                     token_start_index += 1

...             # وبالمثل، قم بالحلقة فوق `word_ids` بدءًا من النهاية للعثور على `end_position` للإجابة
...             for id in word_ids[::-1]:
...                 if id == word_idx_end:
...                     end_position = token_end_index
...                 else:
...                     token_end_index -= 1

...             start_positions.append(start_position)
...             end_positions.append(end_position)

...         else:
...             start_positions.append(cls_index)
...             end_positions.append(cls_index)

...     encoding["image"] = examples["image"]
...     encoding["start_positions"] = start_positions
...     encoding["end_positions"] = end_positions

...     return encoding
```
الآن بعد أن أصبحت لدينا دالة المعالجة هذه، يمكننا تشفير مجموعة البيانات بأكملها:
```py
>>> encoded_train_dataset = dataset_with_ocr["train"].map(
...     encode_dataset, batched=True, batch_size=2, remove_columns=dataset_with_ocr["train"].column_names
... )
>>> encoded_test_dataset = dataset_with_ocr["test"].map(
...     encode_dataset, batched=True, batch_size=2, remove_columns=dataset_with_ocr["test"].column_names
... )
```
دعونا نتحقق من ميزات مجموعة البيانات المشفرة:
```py
>>> encoded_train_dataset.features
{'image': Sequence(feature=Sequence(feature=Sequence(feature=Value(dtype='uint8', id=None), length=-1, id=None), length=-1, id=None), length=-1, id=None),
'input_ids': Sequence(feature=Value(dtype='int32', id=None), length=-1, id=None),
'token_type_ids': Sequence(feature=Value(dtype='int8', id=None), length=-1, id=None),
'attention_mask': Sequence(feature=Value(dtype='int8', id=None), length=-1, id=None),
'bbox': Sequence(feature=Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None), length=-1, id=None),
'start_positions': Value(dtype='int64', id=None),
'end_positions': Value(dtype='int64', id=None)}
```
## التقييم
يتطلب تقييم الإجابة على أسئلة المستندات قدرًا كبيرًا من ما بعد المعالجة. ولتجنب استغراق الكثير من وقتك، يتخطى هذا الدليل خطوة التقييم. لا يزال [مدرب] يحسب خسارة التقييم أثناء التدريب حتى لا تكون في الظلام تمامًا بشأن أداء نموذجك. عادةً ما يتم تقييم الإجابة الاستخراجية على الأسئلة باستخدام F1/exact match.
إذا كنت ترغب في تنفيذها بنفسك، فراجع فصل [الإجابة على الأسئلة](https://huggingface.co/course/chapter7/7?fw=pt#postprocessing) في دورة Hugging Face للاستلهام.

## التدريب
تهانينا! لقد نجحت في اجتياز أصعب جزء من هذا الدليل، والآن أنت مستعد لتدريب نموذجك الخاص.
يتضمن التدريب الخطوات التالية:
* قم بتحميل النموذج باستخدام [`AutoModelForDocumentQuestionAnswering`] باستخدام نفس نقطة التفتيش كما في مرحلة ما قبل المعالجة.
* حدد فرط معلماتك التدريبية في [`TrainingArguments`].
* حدد دالة لدمج الأمثلة معًا، وهنا ستكون [`DefaultDataCollator`] جيدة بما فيه الكفاية
* قم بتمرير الحجج التدريبية إلى [`Trainer`] جنبًا إلى جنب مع النموذج ومجموعة البيانات ومجمع البيانات.
* استدعاء [`~Trainer.train`] لضبط نموذجك بشكل دقيق.
```py
>>> from transformers import AutoModelForDocumentQuestionAnswering

>>> model = AutoModelForDocumentQuestionAnswering.from_pretrained(model_checkpoint)
```
في [`TrainingArguments`]، استخدم `output_dir` لتحديد مكان حفظ نموذجك، وقم بتكوين فرط المعلمات كما تراه مناسبًا.
إذا كنت ترغب في مشاركة نموذجك مع المجتمع، فاضبط `push_to_hub` على `True` (يجب أن تكون قد سجلت الدخول إلى Hugging Face لتحميل نموذجك).
في هذه الحالة، سيكون `output_dir` أيضًا اسم المستودع حيث سيتم دفع نقطة تفتيش النموذج الخاص بك.
```py
>>> from transformers import TrainingArguments

>>> # استبدل هذا بمعرف المستودع الخاص بك
>>> repo_id = "MariaK/layoutlmv2-base-uncased_finetuned_docvqa"

>>> training_args = TrainingArguments(
...     output_dir=repo_id,
...     per_device_train_batch_size=4,
...     num_train_epochs=20,
...     save_steps=200,
...     logging_steps=50,
...     eval_strategy="steps",
...     learning_rate=5e-5,
...     save_total_limit=2,
...     remove_unused_columns=False,
...     push_to_hub=True,
... )
```
قم بتعريف مجمع بيانات بسيط لدمج الأمثلة معًا.
```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator()
```
أخيرًا، قم بجمع كل شيء معًا، واستدعاء [`~Trainer.train`]:
```py
>>> from transformers import Trainer

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     data_collator=data_collator,
...     train_dataset=encoded_train_dataset,
...     eval_dataset=encoded_test_dataset,
...     tokenizer=processor,
... )

>>> trainer.train()
```
لإضافة النموذج النهائي إلى 🤗 Hub، قم بإنشاء بطاقة نموذج واستدعاء `push_to_hub`:
```py
>>> trainer.create_model_card()
>>> trainer.push_to_hub()
```

## الاستنتاج
الآن بعد أن قمت بتدريب نموذج LayoutLMv2 وتحميله إلى 🤗 Hub، يمكنك استخدامه للاستنتاج. أبسط طريقة لتجربة نموذجك المدرب للاستنتاج هي استخدامه في [`Pipeline`].
لنأخذ مثالاً:
```py
>>> example = dataset["test"][2]
>>> question = example["query"]["en"]
>>> image = example["image"]
>>> print(question)
>>> print(example["answers"])
'Who is ‘presiding’ TRRF GENERAL SESSION (PART 1)?'
['TRRF Vice President', 'lee a. waller']
```
بعد ذلك، قم بتنفيذ خط أنابيب للإجابة على أسئلة المستندات باستخدام نموذجك، ومرر مزيج الصورة + السؤال إليه.
```py
>>> from transformers import pipeline

>>> qa_pipeline = pipeline("document-question-answering", model="MariaK/layoutlmv2-base-uncased_finetuned_docvqa")
>>> qa_pipeline(image, question)
[{'score': 0.9949808120727539,
'answer': 'Lee A. Waller',
'start': 55,
'end': 57}]
```
يمكنك أيضًا يدويًا تكرار نتائج خط الأنابيب إذا كنت ترغب في ذلك:
1. خذ صورة وسؤال، وقم بإعدادهما للنموذج باستخدام المعالج من نموذجك.
2. قم بتمرير نتيجة ما قبل المعالجة أو من خلال النموذج.
3. يعيد النموذج `start_logits` و`end_logits`، والتي تشير إلى الرمز الذي يكون في بداية الإجابة والرمز الذي يكون في نهاية الإجابة. كلاهما له شكل (batch_size، sequence_length).
4. خذ argmax على البعد الأخير من كل من `start_logits` و`end_logits` للحصول على `start_idx` المتوقع و`end_idx`.
5. فك تشفير الإجابة باستخدام المعالج.
```py
>>> import torch
>>> from transformers import AutoProcessor
>>> from transformers import AutoModelForDocumentQuestionAnswering

>>> processor = AutoProcessor.from_pretrained("MariaK/layoutlmv2-base-uncased_finetuned_docvqa")
>>> model = AutoModelForDocumentQuestionAnswering.from_pretrained("MariaK/layoutlmv2-base-uncased_finetuned_docvqa")

>>> with torch.no_grad():
...     encoding = processor(image.convert("RGB"), question, return_tensors="pt")
...     outputs = model(**encoding)
...     start_logits = outputs.start_logits
...     end_logits = outputs.end_logits
...     predicted_start_idx = start_logits.argmax(-1).item()
...     predicted_end_idx = end_logits.argmax(-1).item()

>>> processor.tokenizer.decode(encoding.input_ids.squeeze()[predicted_start_idx : predicted_end_idx + 1])
'lee a. waller'
```