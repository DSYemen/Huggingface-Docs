## الإجابة البصرية 

[[open-in-colab]] 

الإجابة البصرية على الأسئلة (VQA) هي مهمة الإجابة على أسئلة مفتوحة بناءً على صورة. وعادة ما يكون المدخل لنماذج دعم هذه المهمة مزيجًا من صورة وسؤال، والمخرج هو إجابة معبر عنها باللغة الطبيعية. 

وفيما يلي بعض أمثلة حالات الاستخدام الجديرة بالملاحظة لـ VQA: 

- تطبيقات الوصول لمساعدة الأفراد ضعاف البصر. 
- التعليم: طرح أسئلة حول المواد المرئية المقدمة في المحاضرات أو الكتب المدرسية. ويمكن أيضًا استخدام VQA في المعارض التفاعلية في المتاحف أو المواقع التاريخية. 
- خدمة العملاء والتجارة الإلكترونية: يمكن لـ VQA تعزيز تجربة المستخدم من خلال السماح للمستخدمين بطرح أسئلة حول المنتجات. 
- استرجاع الصور: يمكن استخدام نماذج VQA لاسترجاع الصور ذات الخصائص المحددة. على سبيل المثال، يمكن للمستخدم أن يسأل "هل هناك كلب؟" للعثور على جميع الصور التي تحتوي على كلاب من مجموعة من الصور. 

في هذا الدليل، ستتعلم كيفية: 

- ضبط نموذج VQA للتصنيف، وتحديدًا [ViLT](../model_doc/vilt)، على مجموعة بيانات [`Graphcore/vqa`](https://huggingface.co/datasets/Graphcore/vqa). 
- استخدام ViLT المضبوط مسبقًا للاستنتاج. 
- تشغيل الاستدلال VQA بدون تدريب مع نموذج توليدي، مثل BLIP-2. 

## ضبط ViLT الدقيق 

يتضمن نموذج ViLT تضمينًا للرموز النصية في محول الرؤية (ViT)، مما يسمح له بتصميم الحد الأدنى لمهام التدريب على الرؤية واللغة (VLP). ويمكن استخدام هذا النموذج لعدة مهام أسفل النهر. وفيما يتعلق بمهمة VQA، يتم وضع رأس مصنف في الأعلى (طبقة خطية في الأعلى من الحالة المخفية النهائية لرموز `[CLS]`) ويتم تهيئتها بشكل عشوائي. وبالتالي، يتم التعامل مع الإجابة على الأسئلة البصرية على أنها **مشكلة تصنيف**. 

وتتعامل النماذج الأحدث، مثل BLIP وBLIP-2 وInstructBLIP، مع VQA كمهمة توليدية. وفيما بعد في هذا الدليل، نوضح كيفية استخدامها للاستدلال VQA بدون تدريب. 

قبل البدء، تأكد من تثبيت جميع المكتبات الضرورية. 

```bash
pip install -q transformers datasets
``` 

ونشجعك على مشاركة نموذجك مع المجتمع. سجل الدخول إلى حساب Hugging Face الخاص بك لتحميله إلى 🤗 Hub. 

عند المطالبة، أدخل رمزك للتسجيل: 

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
``` 

دعنا نحدد نقطة تفتيش النموذج كمتغير عالمي. 

```py
>>> model_checkpoint = "dandelin/vilt-b32-mlm"
``` 

## تحميل البيانات 

لأغراض التوضيح، في هذا الدليل، نستخدم عينة صغيرة جدًا من مجموعة بيانات الإجابة على الأسئلة المرئية المُعلَّمة `Graphcore/vqa`. 

يمكنك العثور على مجموعة البيانات الكاملة على [🤗 Hub](https://huggingface.co/datasets/Graphcore/vqa). 

وكبديل لمجموعة بيانات [`Graphcore/vqa`](https://huggingface.co/datasets/Graphcore/vqa)، يمكنك تنزيل نفس البيانات يدويًا من صفحة مجموعة بيانات VQA الرسمية https://visualqa.org/download.html. إذا كنت تفضل اتباع البرنامج التعليمي باستخدام بياناتك المخصصة، فراجع كيفية [إنشاء مجموعة بيانات صورة](https://huggingface.co/docs/datasets/image_dataset#loading-script) الدليل في وثائق 🤗 Datasets. 

دعنا نحمل أول 200 مثال من الانقسام التحقق من الصحة ونستكشف ميزات مجموعة البيانات: 

```python
>>> from datasets import load_dataset

>>> dataset = load_dataset("Graphcore/vqa", split="validation[:200]")
>>> dataset
Dataset({
features: ['question', 'question_type', 'question_id', 'image_id', 'answer_type', 'label'],
num_rows: 200
})
``` 

دعنا نلقي نظرة على مثال لفهم ميزات مجموعة البيانات: 

```py
>>> dataset[0]
{'question': 'Where is he looking?',
'question_type': 'none of the above',
'question_id': 262148000,
'image_id': '/root/.cache/huggingface/datasets/downloads/extracted/ca733e0e000fb2d7a09fbcc94dbfe7b5a30750681d0e965f8e0a23b1c2f98c75/val2014/COCO_val2014_000000262148.jpg',
'answer_type': 'other',
'label': {'ids': ['at table', 'down', 'skateboard', 'table'],
'weights': [0.30000001192092896,
1.0,
0.30000001192092896,
0.30000001192092896]}}
``` 

وتشمل الميزات ذات الصلة بالمهمة ما يلي: 

- `question`: السؤال الذي يجب الإجابة عليه من الصورة. 
- `image_id`: مسار الصورة التي يشير إليها السؤال. 
- `label`: التعليقات التوضيحية. 

يمكننا إزالة بقية الميزات لأنها لن تكون ضرورية: 

```py
>>> dataset = dataset.remove_columns(['question_type', 'question_id', 'answer_type'])
``` 

كما ترى، تحتوي ميزة `label` على عدة إجابات لنفس السؤال (تسمى `ids` هنا) التي جمعها معلقون بشريون مختلفون. 

ويرجع ذلك إلى أن إجابة السؤال قد تكون ذاتية. في هذه الحالة، كان السؤال "أين ينظر؟". وقد قام بعض الأشخاص بتعليق هذا بـ "down"، بينما علق آخرون بـ "at table"، وعلق شخص آخر بـ "skateboard"، وهكذا. 

الق نظرة على الصورة وفكر في الإجابة التي ستقدمها: 

```python
>>> from PIL import Image

>>> image = Image.open(dataset[0]['image_id'])
>>> image
``` 

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/vqa-example.png" alt="VQA Image Example"/>
</div> 

وبسبب غموض الأسئلة والإجابات، يتم التعامل مع مجموعات البيانات مثل هذه على أنها مشكلة تصنيف متعددة التصنيفات (حيث قد تكون إجابات متعددة صالحة). علاوة على ذلك، بدلاً من إنشاء ترميز ثنائي، يتم إنشاء ترميز ناعم، بناءً على عدد المرات التي ظهر فيها إجابة معينة في التعليقات التوضيحية. 

على سبيل المثال، في المثال أعلاه، لأن الإجابة "down" تم اختيارها بشكل أكبر بكثير من الإجابات الأخرى، فإن لها درجة (تسمى `weight` في مجموعة البيانات) من 1.0، وبقية الإجابات لها درجات <1.0. 

للاحقًا، قم بتهيئة النموذج برأس تصنيف مناسب، دعنا ننشئ قاموسين: أحدهما يقوم بتعيين اسم التصنيف إلى رقم صحيح والعكس صحيح: 

```py
>>> import itertools

>>> labels = [item['ids'] for item in dataset['label']]
>>> flattened_labels = list(itertools.chain(*labels))
>>> unique_labels = list(set(flattened_labels))

>>> label2id = {label: idx for idx, label in enumerate(unique_labels)}
>>> id2label = {idx: label for label, idx in label2id.items()}
``` 

الآن بعد أن حصلنا على الخرائط، يمكننا استبدال الإجابات النصية بمعرفاتها، وتسطيح مجموعة البيانات لمزيد من المعالجة المسبقة. 

```python
>>> def replace_ids(inputs):
...   inputs["label"]["ids"] = [label2id[x] for x in inputs["label"]["ids"]]
...   return inputs


>>> dataset = dataset.map(replace_ids)
>>> flat_dataset = dataset.flatten()
>>> flat_dataset.features
{'question': Value(dtype='string', id=None),
'image_id': Value(dtype='string', id=None),
'label.ids': Sequence(feature=Value(dtype='int64', id=None), length=-1, id=None),
'label.weights': Sequence(feature=Value(dtype='float64', id=None), length=-1, id=None)}
``` 

## البيانات السابقة 

الخطوة التالية هي تحميل معالج ViLT لتحضير بيانات الصورة والنص للنموذج. 

[`ViltProcessor`] يغلف محول BERT ومعالج صور ViLT في معالج واحد مناسب: 

```py
>>> from transformers import ViltProcessor

>>> processor = ViltProcessor.from_pretrained(model_checkpoint)
``` 

لمعالجة البيانات، نحتاج إلى تشفير الصور والأسئلة باستخدام [`ViltProcessor`]. سيستخدم المعالج [`BertTokenizerFast`] لتوكينيز النص وإنشاء `input_ids` و`attention_mask` و`token_type_ids` لبيانات النص. 

أما بالنسبة للصور، فسيستفيد المعالج من [`ViltImageProcessor`] لتصغير حجم الصورة وتطبيعها، وإنشاء `pixel_values` و`pixel_mask`. 

يتم تنفيذ جميع خطوات المعالجة المسبقة هذه تلقائيًا، ولا نحتاج إلا إلى استدعاء `processor`. ومع ذلك، ما زلنا بحاجة إلى إعداد التصنيفات المستهدفة. في هذا التمثيل، يتوافق كل عنصر مع إجابة محتملة (تصنيف). وبالنسبة للإجابات الصحيحة، يحتوي العنصر على درجته (الوزن)، بينما يتم تعيين العناصر المتبقية إلى الصفر. 

تقوم الدالة التالية بتطبيق `processor` على الصور والأسئلة وتنسيق التصنيفات كما هو موضح أعلاه: 

```py
>>> import torch

>>> def preprocess_data(examples):
...     image_paths = examples['image_id']
...     images = [Image.open(image_path) for image_path in image_paths]
...     texts = examples['question']

...     encoding = processor(images, texts, padding="max_length", truncation=True, return_tensors="pt")

...     for k, v in encoding.items():
...           encoding[k] = v.squeeze()

...     targets = []

...     for labels, scores in zip(examples['label.ids'], examples['label.weights']):
...         target = torch.zeros(len(id2label))

...         for label, score in zip(labels, scores):
...             target[label] = score

...         targets.append(target)

...     encoding["labels"] = targets

...     return encoding
``` 

لتطبيق دالة المعالجة المسبقة على مجموعة البيانات بأكملها، استخدم وظيفة 🤗 Datasets [`~datasets.map`]. يمكنك تسريع `map` عن طريق تعيين `batched=True` لمعالجة عناصر متعددة من مجموعة البيانات في نفس الوقت. في هذه المرحلة، لا تتردد في إزالة الأعمدة التي لا تحتاج إليها. 

```py
>>> processed_dataset = flat_dataset.map(preprocess_data, batched=True, remove_columns=['question','question_type',  'question_id', 'image_id', 'answer_type', 'label.ids', 'label.weights'])
>>> processed_dataset
Dataset({
features: ['input_ids', 'token_type_ids', 'attention_mask', 'pixel_values', 'pixel_mask', 'labels'],
num_rows: 200
})
``` 

وكخطوة أخيرة، قم بإنشاء دفعة من الأمثلة باستخدام [`DefaultDataCollator`]: 

```py
>>> from transformers import DefaultDataCollator

>>> data_collator = DefaultDataCollator()
```
بالتأكيد، سأتبع تعليماتك وسأترجم فقط النص الموجود في الفقرات والعناوين:

## تدريب النموذج

الآن أنت مستعد لبدء تدريب نموذجك! قم بتحميل ViLT مع [`ViltForQuestionAnswering`]. حدد عدد التصنيفات إلى جانب تعيينات التصنيف:

في هذه المرحلة، لم يتبق سوى ثلاث خطوات:

1. حدد فرط معلمات التدريب الخاصة بك في [`TrainingArguments`]:

```py
>>> من المحولات استيراد TrainingArguments

 >>> repo_id = "MariaK/vilt_finetuned_200"

 >>> training_args = TrainingArguments(
... train_batch_size=4،
... num_train_epochs=20،
... save_steps=200،
... logging_steps=50،
... learning_rate=5e-5،
... save_total_limit=2،
... remove_unused_columns=False،
... push_to_hub=True،
... )
```

2. قم بتمرير الحجج التدريبية إلى [`Trainer`] جنبًا إلى جنب مع النموذج ومجموعة البيانات والمعالج ومجمع البيانات.

```py
>>> من المحولات استيراد المدرب

 >>> trainer = Trainer(
... model=model،
... args=training_args،
... data_collator=data_collator،
... train_dataset=processed_dataset،
... tokenizer=processor
... )
```

3. استدعاء [`~Trainer.train`] لضبط نموذجك.

```py
>>> trainer.train()
```

بمجرد الانتهاء من التدريب، شارك نموذجك في Hub باستخدام طريقة [`~Trainer.push_to_hub`] لمشاركة نموذجك النهائي على 🤗 Hub:

```py
>>> trainer.push_to_hub()
```

## الاستنتاج

الآن بعد أن ضبطت نموذج ViLT، وقمت بتحميله إلى 🤗 Hub، يمكنك استخدامه للاستنتاج. أسهل طريقة لتجربة نموذجك المُدرب للاستنتاج هي استخدامه في [`Pipeline`].

```py
>>> من المحولات استيراد الأنابيب

 >>> pipe = pipeline("visual-question-answering"، model="MariaK/vilt_finetuned_200")
```

تم تدريب النموذج في هذا الدليل على 200 مثال فقط، لذا لا تتوقع الكثير منه. دعونا نرى ما إذا كان قد تعلم شيئًا على الأقل من البيانات وخذ المثال الأول من مجموعة البيانات لتوضيح الاستدلال:

```py
>>> example = dataset[0]
>>> image = Image.open(example['image_id'])
>>> question = example['question']
>>> print(question)
>>> pipe(image، question، top_k=1)
"أين ينظر؟"
[{'score': 0.5498199462890625، 'answer': 'down'}]
```

على الرغم من عدم الثقة، إلا أن النموذج قد تعلم بالفعل شيئًا ما. مع المزيد من الأمثلة وفترات التدريب الأطول، ستحصل على نتائج أفضل بكثير!

يمكنك أيضًا إعادة إنتاج نتائج الأنبوب يدويًا إذا أردت:

1. خذ صورة وسؤال، وقم بإعدادهما للنموذج باستخدام المعالج من نموذجك.

2. قم بتمرير نتيجة ما قبل المعالجة أو من خلال النموذج.

3. من logits، احصل على معرف الإجابة الأكثر احتمالًا، واعثر على الإجابة الفعلية في `id2label`.

```py
>>> processor = ViltProcessor.from_pretrained("MariaK/vilt_finetuned_200")

>>> image = Image.open(example['image_id'])
>>> question = example['question']

>>> # إعداد المدخلات
>>> المدخلات = المعالج (الصورة، السؤال، return_tensors="pt")

>>> model = ViltForQuestionAnswering.from_pretrained("MariaK/vilt_finetuned_200")

>>> # تمرير إلى الأمام
>>> مع الشعلة. لا_grad ():
... إخراج = النموذج (** المدخلات)

>>> logits = outputs.logits
>>> idx = logits.argmax (-1).item ()
>>> print ("الإجابة المتوقعة:"، model.config.id2label [idx])
الإجابة المتوقعة: أسفل
```

## VQA بدون تصوير

عامل النموذج السابق VQA كمهام تصنيف. تعامل بعض النماذج الحديثة، مثل BLIP وBLIP-2 وInstructBLIP، مع VQA كمهمة توليدية. دعنا نأخذ [BLIP-2](../model_doc/blip-2) كمثال. لقد قدم نموذجًا جديدًا للتعلم التلقائي للرؤية واللغة حيث يمكن استخدام أي مزيج من مشفر الرؤية المسبق التدريب وLLM (تعرف على المزيد في منشور المدونة [BLIP-2](https://huggingface.co/blog/blip-2)).

يتيح ذلك تحقيق نتائج رائدة على مستوى المهام المرئية اللغوية المتعددة بما في ذلك الإجابة على الأسئلة المرئية.

دعونا نوضح كيف يمكنك استخدام هذا النموذج لـ VQA. أولاً، دعنا نحمل النموذج. هنا سنرسل النموذج صراحة إلى وحدة معالجة الرسوميات، إذا كانت متوفرة، والتي لم نكن بحاجة إلى القيام بها سابقًا عند التدريب، حيث يتعامل [`Trainer`] مع ذلك تلقائيًا:

```py
>>> من المحولات استيراد AutoProcessor، Blip2ForConditionalGeneration
>>> استيراد الشعلة

>>> المعالج = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
>>> model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b"، torch_dtype=torch.float16)
>>> الجهاز = "cuda" إذا كان torch.cuda.is_available () else "cpu"
>>> model.to (الجهاز)
```

يأخذ النموذج الصورة والنص كإدخال، لذا دعنا نستخدم نفس زوج الصورة/السؤال بالضبط من المثال الأول في مجموعة بيانات VQA:

```py
>>> example = dataset[0]
>>> image = Image.open(example['image_id'])
>>> question = example['question']
```

لاستخدام BLIP-2 لمهمة الإجابة على الأسئلة المرئية، يجب أن يتبع النص الفوري تنسيقًا محددًا: `Question: {} Answer:`.

```py
>>> الفوري = "السؤال: {} الإجابة:"
```

الآن نحن بحاجة إلى معالجة الصورة/الفوري مع المعالج الخاص بالنموذج، وتمرير الإدخال المعالج عبر النموذج، وفك تشفير الإخراج:

```py
>>> المدخلات = المعالج (الصورة، النص = الفوري، return_tensors="pt").to (الجهاز، torch.float16)

>>> generated_ids = model.generate (** المدخلات، max_new_tokens=10)
>>> generated_text = processor.batch_decode (generated_ids، skip_special_tokens=True) [0].strip ()
>>> print (generated_text)
"إنه ينظر إلى الحشد"
```

كما ترون، تعرف النموذج على الحشد واتجاه الوجه (ينظر إلى الأسفل)، ومع ذلك، يبدو أنه يغفل عن حقيقة أن الحشد موجود خلف المتزلج. ومع ذلك، في الحالات التي يكون من غير العملي فيها الحصول على مجموعات بيانات بشرية موسومة، يمكن أن ينتج هذا النهج نتائج مفيدة بسرعة.