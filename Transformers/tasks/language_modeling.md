# النمذجة اللغوية السببية

هناك نوعان من النمذجة اللغوية، النمذجة السببية والنمذجة المقنعة. يوضح هذا الدليل النمذجة اللغوية السببية.

تُستخدم نماذج اللغة السببية بشكل متكرر لتوليد النصوص. يمكنك استخدام هذه النماذج لتطبيقات إبداعية مثل اختيار مغامرتك النصية الخاصة أو مساعد ترميز ذكي مثل Copilot أو CodeParrot.

تتنبأ النمذجة اللغوية السببية بالرمز التالي في تسلسل الرموز، ولا يمكن للنموذج سوى الاهتمام بالرموز الموجودة على اليسار. وهذا يعني أن النموذج لا يمكنه رؤية الرموز المستقبلية. يعد GPT-2 مثالًا على نموذج اللغة السببي.

سيوضح هذا الدليل كيفية:

1. ضبط نموذج DistilGPT2 الدقيق على مجموعة فرعية من r/askscience من مجموعة بيانات ELI5.
2. استخدام نموذجك الدقيق للاستنتاج.

قبل أن تبدأ، تأكد من تثبيت جميع المكتبات الضرورية:

```bash
pip install transformers datasets evaluate
```

نحن نشجعك على تسجيل الدخول إلى حساب Hugging Face الخاص بك حتى تتمكن من تحميل نموذجك ومشاركته مع المجتمع. عندما يُطلب منك ذلك، أدخل رمزك للتسجيل:

```py
>>> from huggingface_hub import notebook_login

>>> notebook_login()
```

## تحميل مجموعة بيانات ELI5

ابدأ بتحميل أول 5000 مثال من مجموعة بيانات [ELI5-Category](https://huggingface.co/datasets/eli5_category) باستخدام مكتبة Datasets 🤗. سيعطيك هذا فرصة للتجربة والتأكد من أن كل شيء يعمل قبل قضاء المزيد من الوقت في التدريب على مجموعة البيانات الكاملة.

```py
>>> from datasets import load_dataset

>>> eli5 = load_dataset("eli5_category", split="train[:5000]")
```

قسِّم مجموعة البيانات إلى مجموعات فرعية للتدريب والاختبار باستخدام طريقة [`~datasets.Dataset.train_test_split`]:

```py
>>> eli5 = eli5.train_test_split(test_size=0.2)
```

ثم الق نظرة على مثال:

```py
>>> eli5["train"][0]
{'q_id': '7h191n',
'title': 'ما الذي يعنيه مشروع قانون الضرائب الذي تم تمريره اليوم؟ كيف سيؤثر على الأمريكيين في كل شريحة ضريبية؟',
'selftext': '',
'category': 'اقتصاد',
'subreddit': 'explainlikeimfive',
'answers': {'a_id': ['dqnds8l', 'dqnd1jl', 'dqng3i1', 'dqnku5x'],
'النص': ["مشروع قانون الضرائب عبارة عن 500 صفحة وكانت هناك الكثير من التغييرات التي لا تزال جارية حتى النهاية. الأمر لا يقتصر على تعديل شرائح ضريبة الدخل، بل هو مجموعة كاملة من التغييرات. وبالتالي لا يوجد إجابة جيدة على سؤالك. النتائج الرئيسية هي: - انخفاض كبير في معدل ضريبة الشركات سيجعل الشركات الكبرى سعيدة للغاية. - سيؤدي تغيير معدل المرور إلى جعل بعض أنماط الأعمال (شركات القانون وصناديق التحوط) سعيدة للغاية - تعديلات ضريبة الدخل معتدلة، ومن المقرر أن تنتهي (على الرغم من أنها النوع الذي قد يتم إعادة تطبيقه دائمًا دون جعله دائمًا) - يخسر الأشخاص في الولايات ذات الضرائب المرتفعة (كاليفورنيا ونيويورك)، وقد ينتهي الأمر بزيادة الضرائب.",
'لا شيء حتى الآن. يجب التوفيق بينه وبين مشروع قانون مجلس النواب المختلف تمامًا ثم تمريره مرة أخرى.',
'أيضا: هل ينطبق هذا على ضرائب 2017؟ أم أنها تبدأ بضرائب 2018؟',
'توضح هذه المقالة كلا من مشروعي مجلس النواب والشيوخ، بما في ذلك التغييرات المقترحة على ضرائب الدخل الخاصة بك بناءً على مستوى دخلك. URL_0'],
'الحساب': [21، 19، 5، 3]،
'نص_العناوين': [[],
[]،
[]،
['https://www.investopedia.com/news/trumps-tax-reform-what-can-be-done/']]},
'العناوين_العناوين': ['url'],
'selftext_urls': ['url']}
```

على الرغم من أن هذا قد يبدو كثيرًا، إلا أنك مهتم حقًا بحقل "النص". ما هو رائع حول مهام النمذجة اللغوية هو أنك لا تحتاج إلى تسميات (تُعرف أيضًا باسم المهمة غير الخاضعة للإشراف) لأن الكلمة التالية *هي* التسمية.

## معالجة مسبقة

الخطوة التالية هي تحميل برنامج تشفير DistilGPT2 لمعالجة حقل "النص":

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilgpt2")
```

ستلاحظ من المثال أعلاه، أن حقل "النص" موجود بالفعل داخل "الإجابات". وهذا يعني أنك ستحتاج إلى استخراج حقل "النص" من هيكله المضمن باستخدام طريقة ["flatten"](https://huggingface.co/docs/datasets/process#flatten):

```py
>>> eli5 = eli5.flatten()
>>> eli5["train"][0]
{'q_id': '7h191n',
'title': 'ما الذي يعنيه مشروع قانون الضرائب الذي تم تمريره اليوم؟ كيف سيؤثر على الأمريكيين في كل شريحة ضريبية؟',
'selftext': '',
'category': 'اقتصاد',
'subreddit': 'explainlikeimfive',
'answers.a_id': ['dqnds8l', 'dqnd1jl', 'dqng3i1', 'dqnku5x'],
'answers.text': ["مشروع قانون الضرائب عبارة عن 500 صفحة وكانت هناك الكثير من التغييرات التي لا تزال جارية حتى النهاية. الأمر لا يقتصر على تعديل شرائح ضريبة الدخل، بل هو مجموعة كاملة من التغييرات. وبالتالي لا يوجد إجابة جيدة على سؤالك. النتائج الرئيسية هي: - انخفاض كبير في معدل ضريبة الشركات سيجعل الشركات الكبرى سعيدة للغاية. - سيؤدي تغيير معدل المرور إلى جعل بعض أنماط الأعمال (شركات القانون وصناديق التحوط) سعيدة للغاية - تعديلات ضريبة الدخل معتدلة، ومن المقرر أن تنتهي (على الرغم من أنها النوع الذي قد يتم إعادة تطبيقه دائمًا دون جعله دائمًا) - يخسر الأشخاص في الولايات ذات الضرائب المرتفعة (كاليفورنيا ونيويورك)، وقد ينتهي الأمر بزيادة الضرائب.",
'لا شيء حتى الآن. يجب التوفيق بينه وبين مشروع قانون مجلس النواب المختلف تمامًا ثم تمريره مرة أخرى.',
'أيضا: هل ينطبق هذا على ضرائب 2017؟ أم أنها تبدأ بضرائب 2018؟',
'توضح هذه المقالة كلا من مشروعي مجلس النواب والشيوخ، بما في ذلك التغييرات المقترحة على ضرائب الدخل الخاصة بك بناءً على مستوى دخلك. URL_0'],
'answers.score': [21, 19, 5, 3],
'answers.text_urls': [[],
[]،
[]،
['https://www.investopedia.com/news/trumps-tax-reform-what-can-be-done/']],
'العناوين_العناوين': ['url'],
'selftext_urls': ['url']}
```

كل حقل فرعي هو الآن عمود منفصل كما هو موضح بالبادئة "الإجابات"، وحقل "النص" هو قائمة الآن. بدلاً من تشفير كل جملة بشكل منفصل، قم بتحويل القائمة إلى سلسلة حتى تتمكن من تشفيرها بشكل مشترك.

هذه هي دالة المعالجة المسبقة الأولى لدمج قائمة السلاسل لكل مثال وتشفير النتيجة:

```py
>>> def preprocess_function(examples):
...     return tokenizer([" ".join(x) for x in examples["answers.text"]])
```

لتطبيق دالة المعالجة المسبقة هذه على مجموعة البيانات بأكملها، استخدم طريقة [`~datasets.Dataset.map`] في مكتبة Datasets 🤗. يمكنك تسريع وظيفة "المخطط" عن طريق تعيين "batched=True" لمعالجة عناصر متعددة من مجموعة البيانات في وقت واحد، وزيادة عدد العمليات باستخدام "num_proc". احذف أي أعمدة لا تحتاجها:

```py
>>> tokenized_eli5 = eli5.map(
...     preprocess_function,
...     batched=True,
...     num_proc=4,
...     remove_columns=eli5["train"].column_names,
... )
```

تحتوي مجموعة البيانات هذه على تسلسلات الرموز، ولكن بعضها أطول من طول الإدخال الأقصى للنموذج.

الآن يمكنك استخدام دالة المعالجة المسبقة الثانية ل:

- دمج جميع التسلسلات
- تقسيم التسلسلات المدمجة إلى قطع أقصر محددة بواسطة "block_size"، والتي يجب أن تكون أقصر من طول الإدخال الأقصى وقصيرة بدرجة كافية لذاكرة GPU RAM.

```py
>>> block_size = 128


>>> def group_texts(examples):
...     # Concatenate all texts.
...     concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
...     total_length = len(concatenated_examples[list(examples.keys())[0]])
...     # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
...     # customize this part to your needs.
...     if total_length >= block_size:
...         total_length = (total_length // block_size) * block_size
...     # Split by chunks of block_size.
...     result = {
...         k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
...         for k, t in concatenated_examples.items()
...     }
...     result["labels"] = result["input_ids"].copy()
...     return result
```

طبق وظيفة "group_texts" على مجموعة البيانات بأكملها:

```py
>>> lm_dataset = tokenized_eli5.map(group_texts, batched=True, num_proc=4)
```

الآن قم بإنشاء دفعة من الأمثلة باستخدام [`DataCollatorForLanguageModeling`]. من الأكثر كفاءة *حشو* الجمل ديناميكيًا إلى أطول طول في دفعة أثناء التجميع، بدلاً من حشو مجموعة البيانات بأكملها إلى الطول الأقصى.

<frameworkcontent>
<pt>
استخدم رمز نهاية التسلسل كرموز حشو وقم بتعيين "mlm=False". سيتم استخدام هذا كمدخلات كعلامات منزاحة إلى اليمين بواسطة عنصر واحد:

```py
>>> from transformers import DataCollatorForLanguageModeling

>>> tokenizer.pad_token = tokenizer.eos_token
>>> data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
```

</pt>
<tf>
استخدم رمز نهاية التسلسل كرموز حشو وقم بتعيين "mlm=False". سيتم استخدام هذا كمدخلات كعلامات منزاحة إلى اليمين بواسطة عنصر واحد:

```py
>>> from transformers import DataCollatorForLanguageModeling

>>> data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False، return_tensors="tf")
```

</tf>
</frameworkcontent>
بالتأكيد، سأتبع تعليماتك بدقة. فيما يلي ترجمة النص الموجود في الفقرات والعناوين:

## التدريب

إذا لم تكن معتادًا على ضبط نموذج باستخدام [`Trainer`]، فراجع [البرنامج التعليمي الأساسي] (../training#train-with-pytorch-trainer)

أنت الآن على استعداد لبدء تدريب نموذجك! قم بتحميل DistilGPT2 باستخدام [`AutoModelForCausalLM`]:

```python
>>> from transformers import AutoModelForCausalLM, TrainingArguments, Trainer

>>> model = AutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
```

في هذه المرحلة، لم يتبق سوى ثلاث خطوات:

1. حدد فرط معلمات التدريب الخاصة بك في [`TrainingArguments`]. المعلمة المطلوبة الوحيدة هي `output_dir` التي تحدد أين يتم حفظ نموذجك. ستقوم بدفع هذا النموذج إلى المحور عن طريق تعيين `push_to_hub=True` (يجب أن تكون قد سجلت الدخول إلى Hugging Face لتحميل نموذجك).

2. مرر فرط معلمات التدريب إلى [`Trainer`] جنبًا إلى جنب مع النموذج ومجموعات البيانات ومجمع البيانات.

3. استدعاء [`~Trainer.train`] لضبط نموذجك.

```python
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_eli5_clm-model",
...     eval_strategy="epoch"،
...     learning_rate=2e-5,
...     weight_decay=0.01,
...     push_to_hub=True,
... )

>>> trainer = Trainer(
...     model=model,
...     args=training_args,
...     train_dataset=lm_dataset["train"]،
...     eval_dataset=lm_dataset["test"]،
...     data_collator=data_collator,
... )

>>> trainer.train()
```

بمجرد اكتمال التدريب، استخدم طريقة [`~transformers.Trainer.evaluate`] لتقييم نموذجك والحصول على غموضه:

```python
>>> import math

>>> eval_results = trainer.evaluate()
>>> print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")
Perplexity: 49.61
```

بعد ذلك، شارك نموذجك في المحور باستخدام طريقة [`~transformers.Trainer.push_to_hub`] حتى يتمكن الجميع من استخدام نموذجك:

```python
>>> trainer.push_to_hub()
```

إذا لم تكن معتادًا على ضبط نموذج باستخدام Keras، فراجع [البرنامج التعليمي الأساسي] (../training#train-a-tensorflow-model-with-keras)

لضبط نموذج في TensorFlow، ابدأ بإعداد دالة محسن ومعدل تعلم وجدول، وبعض فرط معلمات التدريب:

```python
>>> from transformers import create_optimizer, AdamWeightDecay

>>> optimizer = AdamWeightDecay(learning_rate=2e-5, weight_decay_rate=0.01)
```

بعد ذلك، يمكنك تحميل DistilGPT2 باستخدام [`TFAutoModelForCausalLM`]:

```python
>>> from transformers import TFAutoModelForCausalLM

>>> model = TFAutoModelForCausalLM.from_pretrained("distilbert/distilgpt2")
```

قم بتحويل مجموعات البيانات الخاصة بك إلى تنسيق `tf.data.Dataset` باستخدام [`~transformers.TFPreTrainedModel.prepare_tf_dataset`]:

```python
>>> tf_train_set = model.prepare_tf_dataset(
...     lm_dataset["train"]،
...     shuffle=True,
...     batch_size=16,
...     collate_fn=data_collator,
... )

>>> tf_test_set = model.prepare_tf_dataset(
...     lm_dataset["test"]،
...     shuffle=False,
...     batch_size=16,
...     collate_fn=data_collator,
... )
```

قم بتكوين النموذج للتدريب مع [`compile`] (https://keras.io/api/models/model_training_apis/#compile-method). لاحظ أن جميع نماذج Transformers لديها دالة خسارة ذات صلة بالمهمة بشكل افتراضي، لذلك لا تحتاج إلى تحديد واحدة ما لم ترغب في ذلك:

```python
>>> import tensorflow as tf

>>> model.compile(optimizer=optimizer) # لا توجد حجة الخسارة!
```

يمكن القيام بذلك عن طريق تحديد المكان الذي ستدفع فيه نموذجك ومعالج النصوص في [`~transformers.PushToHubCallback`]:

```python
>>> from transformers.keras_callbacks import PushToHubCallback

>>> callback = PushToHubCallback(
...     output_dir="my_awesome_eli5_clm-model"،
...     tokenizer=tokenizer,
... )
```

أخيرًا، أنت على استعداد لبدء تدريب نموذجك! استدعاء [`fit`] (https://keras.io/api/models/model_training_apis/#fit-method) مع مجموعات البيانات التدريبية والتحقق من صحتها، وعدد العصور، واستدعاء العودة الخاصة بك لضبط النموذج:

```python
>>> model.fit(x=tf_train_set, validation_data=tf_test_set, epochs=3, callbacks=[callback])
```

بمجرد اكتمال التدريب، يتم تحميل نموذجك تلقائيًا إلى المحور حتى يتمكن الجميع من استخدامه!

للحصول على مثال أكثر تفصيلاً حول كيفية ضبط نموذج للنمذجة اللغوية السببية، راجع الدفتر المقابل
[دفتر ملاحظات PyTorch] (https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb)
أو [دفتر TensorFlow] (https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb).

## الاستنتاج

رائع، الآن بعد أن ضبطت نموذجًا، يمكنك استخدامه للاستنتاج!

فكر في موجه ترغب في توليد نص منه:

```python
>>> prompt = "تسمح الطفرة الجسدية لنظام المناعة"
```

أبسط طريقة لتجربة نموذجك المضبوط للاستنتاج هي استخدامه في [`pipeline`]. قم بتنفيذ مثيل `pipeline` لتوليد النص باستخدام نموذجك، ومرر نصك إليه:

```python
>>> from transformers import pipeline

>>> generator = pipeline("text-generation", model="username/my_awesome_eli5_clm-model")
>>> generator(prompt)
[{'generated_text': 'تسمح الطفرة الجسدية لنظام المناعة بالتعامل مع الأدوية بالقدرة على التكيف مع وضع بيئي مختلف.\n\n\nإن الضرر الذي يلحق بالعدوى ناتج عن قدرة نظام المناعة على أداء مهامه الخاصة بالتصحيح الذاتي. "}]
```

قم برمز النص وإرجاع `input_ids` كتوترات PyTorch:

```python
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("username/my_awesome_eli5_clm-model")
>>> inputs = tokenizer(prompt, return_tensors="pt").input_ids
```

استخدم طريقة [`~generation.GenerationMixin.generate`] لتوليد النص.
لمزيد من التفاصيل حول استراتيجيات توليد النص المختلفة والمعلمات للتحكم في التوليد، راجع صفحة [استراتيجيات توليد النص] (../generation_strategies).

```python
>>> from transformers import AutoModelForCausalLM

>>> model = AutoModelForCausalLM.from_pretrained("username/my_awesome_eli5_clm-model")
>>> outputs = model.generate(inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
```

قم بفك رموز رموز الرموز المولدة مرة أخرى إلى نص:

```python
>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
["تسمح الطفرة الجسدية لنظام المناعة بالتعامل مع الأدوية بالقدرة على التكيف مع وضع بيئي مختلف. وبعبارة أخرى، يمكن لنظام "الطفرة" مساعدة نظام المناعة على التكيف مع وضع بيئي مختلف أو في بعض الحالات حتى حياة واحدة. على النقيض من ذلك، وجد الباحثون في جامعة ماساتشوستس في بوسطن أن "الطفرة" أقوى بكثير في الفئران منها في البشر ولكن يمكن العثور عليها في البشر، وأنها ليست مجهولة تمامًا لنظام المناعة. دراسة حول كيفية نظام المناعة"]
```

قم برمز النص وإرجاع `input_ids` كتوترات TensorFlow:

```python
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("username/my_awesome_eli5_clm-model")
>>> inputs = tokenizer(prompt, return_tensors="tf").input_ids
```

استخدم طريقة [`~transformers.generation_tf_utils.TFGenerationMixin.generate`] لإنشاء الملخص. لمزيد من التفاصيل حول استراتيجيات توليد النص المختلفة والمعلمات للتحكم في التوليد، راجع صفحة [استراتيجيات توليد النص] (../generation_strategies).

```python
>>> from transformers import TFAutoModelForCausalLM

>>> model = TFAutoModelForCausalLM.from_pretrained("username/my_awesome_eli5_clm-model")
>>> outputs = model.generate(input_ids=inputs, max_new_tokens=100, do_sample=True, top_k=50, top_p=0.95)
```

قم بفك رموز رموز الرموز المولدة مرة أخرى إلى نص:

```python
>>> tokenizer.batch_decode(outputs, skip_special_tokens=True)
['تسمح الطفرة الجسدية لنظام المناعة بالكشف عن وجود فيروسات أخرى مع زيادة انتشارها. لذلك، حدد الباحثون نسبة عالية من الفيروسات البشرية. تزداد نسبة الفيروسات المرتبطة بالفيروسات في دراستنا مع تقدم العمر. لذلك، نقترح خوارزمية بسيطة للكشف عن وجود هذه الفيروسات الجديدة في عيناتنا كعلامة على تحسن المناعة. تهدف دراسة أولى قائمة على هذا الخوارزمية، والتي ستنشر في مجلة Science يوم الجمعة، إلى إظهار أن هذا الاكتشاف يمكن أن يترجم إلى تطوير لقاح أفضل يكون أكثر فعالية']
```