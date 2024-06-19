## النمذجة اللغوية المقنعة

تتنبأ النمذجة اللغوية المقنعة برمز مقنع في تسلسل، ويمكن للنموذج الاهتمام بالرموز ثنائي الاتجاه. وهذا يعني أن النموذج لديه إمكانية الوصول الكامل إلى الرموز الموجودة على اليسار واليمين. تعد النمذجة اللغوية المقنعة رائعة للمهام التي تتطلب فهمًا سياقيًا جيدًا لتسلسل كامل. BERT هو مثال على نموذج اللغة المقنع.

سيوضح هذا الدليل كيفية:

1. ضبط نموذج [DistilRoBERTa](https://huggingface.co/distilbert/distilroberta-base) الدقيق على الجزء الفرعي [r/askscience](https://www.reddit.com/r/askscience/) لمجموعة بيانات [ELI5](https://huggingface.co/datasets/eli5).
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

ابدأ بتحميل أول 5000 مثال من مجموعة بيانات [ELI5-Category](https://huggingface.co/datasets/eli5_category) باستخدام مكتبة Datasets 🤗. سيتيح لك هذا فرصة التجربة والتأكد من أن كل شيء يعمل قبل قضاء المزيد من الوقت في التدريب على مجموعة البيانات الكاملة.

```py
>>> from datasets import load_dataset

>>> eli5 = load_dataset("eli5_category", split="train[:5000]")
```

قسِّم مجموعة البيانات إلى مجموعتين فرعيتين للتدريب والاختبار باستخدام طريقة [`~datasets.Dataset.train_test_split`]:

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
'text': ["مشروع قانون الضرائب عبارة عن 500 صفحة وكانت هناك الكثير من التغييرات التي لا تزال مستمرة حتى النهاية. الأمر لا يقتصر على تعديل الشرائح الضريبية على الدخل، بل هو مجموعة كاملة من التغييرات. وبالتالي لا يوجد إجابة جيدة على سؤالك. النقاط الرئيسية هي: - هناك تخفيض كبير في معدل ضريبة الشركات مما سيجعل الشركات الكبرى سعيدة للغاية. - سيؤدي تغيير معدل المرور إلى سعادة بعض أنماط الأعمال (مكاتب المحاماة، وصناديق التحوط) بشكل كبير - تعديلات الدخل الضريبي معتدلة، ومن المقرر أن تنتهي (على الرغم من أنها النوع الذي قد يتم إعادة تطبيقه دائمًا دون جعله دائمًا) - يخسر الأشخاص في الولايات ذات الضرائب المرتفعة (كاليفورنيا، نيويورك)، وقد ينتهي بهم الأمر برفع الضرائب.",
'لم يتم بعد. يجب التوفيق بينه وبين مشروع قانون مجلس النواب المختلف تمامًا ثم إقراره مرة أخرى.',
'أيضا: هل ينطبق هذا على ضرائب عام 2017؟ أم أنه يبدأ بضرائب عام 2018؟',
'توضح هذه المقالة كلا من مشروعي مجلسي النواب والشيوخ، بما في ذلك التغييرات المقترحة على ضرائب الدخل الخاصة بك بناءً على مستوى دخلك. URL_0'],
'score': [21، 19، 5، 3]،
'text_urls': [[],
[]،
[]،
['https://www.investopedia.com/news/trumps-tax-reform-what-can-be-done/']]},
'title_urls': ['url'],
'selftext_urls': ['url']}
```

على الرغم من أن هذا قد يبدو كثيرًا، إلا أنك مهتم حقًا بحقل `النص`. ما هو رائع حول مهام نمذجة اللغة هو أنك لا تحتاج إلى تسميات (تُعرف أيضًا باسم المهمة غير الخاضعة للإشراف) لأن الكلمة التالية *هي* التسمية.

## معالجة مسبقة

بالنسبة للنمذجة اللغوية المقنعة، تتمثل الخطوة التالية في تحميل برنامج Tokenizer DistilRoBERTa لمعالجة حقل `النص` الفرعي:

```py
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained("distilbert/distilroberta-base")
```

ستلاحظ من المثال أعلاه، أن حقل `النص` موجود بالفعل داخل `الإجابات`. وهذا يعني أنك ستحتاج إلى استخراج حقل `النص` من هيكله المضمن باستخدام طريقة [`flatten`](https://huggingface.co/docs/datasets/process#flatten):

```py
>>> eli5 = eli5.flatten()
>>> eli5["train"][0]
{'q_id': '7h191n',
'title': 'ما الذي يعنيه مشروع قانون الضرائب الذي تم تمريره اليوم؟ كيف سيؤثر على الأمريكيين في كل شريحة ضريبية؟',
'selftext': '',
'category': 'اقتصاد',
'subreddit': 'explainlikeimfive',
'answers.a_id': ['dqnds8l', 'dqnd1jl', 'dqng3i1', 'dqnku5x'],
'answers.text': ["مشروع قانون الضرائب عبارة عن 500 صفحة وكانت هناك الكثير من التغييرات التي لا تزال مستمرة حتى النهاية. الأمر لا يقتصر على تعديل الشرائح الضريبية على الدخل، بل هو مجموعة كاملة من التغييرات. وبالتالي لا يوجد إجابة جيدة على سؤالك. النقاط الرئيسية هي: - هناك تخفيض كبير في معدل ضريبة الشركات مما سيجعل الشركات الكبرى سعيدة للغاية. - سيؤدي تغيير معدل المرور إلى سعادة بعض أنماط الأعمال (مكاتب المحاماة، وصناديق التحوط) بشكل كبير - تعديلات الدخل الضريبي معتدلة، ومن المقرر أن تنتهي (على الرغم من أنها النوع الذي قد يتم إعادة تطبيقه دائمًا دون جعله دائمًا) - يخسر الأشخاص في الولايات ذات الضرائب المرتفعة (كاليفورنيا، نيويورك)، وقد ينتهي بهم الأمر برفع الضرائب.",
'لم يتم بعد. يجب التوفيق بينه وبين مشروع قانون مجلس النواب المختلف تمامًا ثم إقراره مرة أخرى.',
'أيضا: هل ينطبق هذا على ضرائب عام 2017؟ أم أنه يبدأ بضرائب عام 2018؟',
'توضح هذه المقالة كلا من مشروعي مجلسي النواب والشيوخ، بما في ذلك التغييرات المقترحة على ضرائب الدخل الخاصة بك بناءً على مستوى دخلك. URL_0'],
'answers.score': [21، 19، 5، 3]،
'answers.text_urls': [[],
[]،
[]،
['https://www.investopedia.com/news/trumps-tax-reform-what-can-be-done/']],
'title_urls': ['url'],
'selftext_urls': ['url']}
```

كل حقل فرعي هو الآن عمود منفصل كما هو موضح بالتسمية `الإجابات`، وحقل `النص` هو قائمة الآن. بدلاً من توكينيز كل جملة بشكل منفصل، قم بتحويل القائمة إلى سلسلة حتى تتمكن من توكينيزها بشكل مشترك.

هذه هي دالة المعالجة المسبقة الأولى لدمج قائمة السلاسل لكل مثال وتوكنيز النتيجة:

```py
>>> def preprocess_function(examples):
...     return tokenizer([" ".join(x) for x in examples["answers.text"])]
```

لتطبيق دالة المعالجة المسبقة هذه على مجموعة البيانات بأكملها، استخدم طريقة [`~datasets.Dataset.map`] في مكتبة Datasets 🤗. يمكنك تسريع وظيفة `map` عن طريق تعيين `batched=True` لمعالجة عناصر متعددة من مجموعة البيانات في وقت واحد، وزيادة عدد العمليات باستخدام `num_proc`. احذف أي أعمدة لا تحتاجها:

```py
>>> tokenized_eli5 = eli5.map(
...     preprocess_function،
...     batched=True،
...     num_proc=4،
...     remove_columns=eli5["train"].column_names،
... )
```

تحتوي مجموعة البيانات هذه على تسلسلات الرموز، ولكن بعضها أطول من طول الإدخال الأقصى للنموذج.

الآن يمكنك استخدام دالة المعالجة المسبقة الثانية ل:

- دمج جميع التسلسلات
- تقسيم التسلسلات المدمجة إلى قطع أقصر محددة بواسطة `block_size`، والتي يجب أن تكون أقصر من طول الإدخال الأقصى وقصيرة بدرجة كافية لذاكرة GPU RAM.

```py
>>> block_size = 128


>>> def group_texts(examples):
...     # دمج جميع النصوص.
...     concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
...     total_length = len(concatenated_examples[list(examples.keys())[0]])
...     # نتخلص من الباقي الصغير، ويمكننا إضافة وسادة إذا كان النموذج يدعمها بدلاً من هذا الانخفاض، يمكنك
...     # قم بتخصيص هذا الجزء وفقًا لاحتياجاتك.
...     if total_length >= block_size:
...         total_length = (total_length // block_size) * block_size
...     # تقسيم حسب كتل من block_size.
...     result = {
...         k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
...         for k, t in concatenated_examples.items()
...     }
...     return result
```

قم بتطبيق وظيفة `group_texts` على مجموعة البيانات بأكملها:

```py
>>> lm_dataset = tokenized_eli5.map(group_texts، batched=True، num_proc=4)
```

الآن قم بإنشاء دفعة من الأمثلة باستخدام [`DataCollatorForLanguageModeling`]. من الأكثر كفاءة *حشو* الجمل ديناميكيًا إلى أطول طول في دفعة أثناء التجميع، بدلاً من حشو مجموعة البيانات بأكملها إلى الطول الأقصى.

<frameworkcontent>
<pt>

استخدم رمز نهاية التسلسل كرموز حشو وحدد `mlm_probability` لإخفاء الرموز عشوائيًا كلما قمت بالتنقل خلال البيانات:

```py
>>> from transformers import DataCollatorForLanguageModeling

>>> tokenizer.pad_token = tokenizer.eos_token
>>> data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
```

</pt>
<tf>

استخدم رمز نهاية التسلسل كرموز حشو وحدد `mlm_probability` لإخفاء الرموز عشوائيًا كلما قمت بالتنقل خلال البيانات:

```py
>>> from transformers import DataCollatorForLanguageModeling

>>> data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15, return_tensors="tf")
```

</tf>

</frameworkcontent>
## التدريب

إذا لم تكن معتاداً على ضبط نموذج باستخدام [`Trainer`] ، فراجع البرنامج التعليمي الأساسي [هنا] (../training # train-with-pytorch-trainer) !

أنت الآن على استعداد لبدء تدريب نموذجك! قم بتحميل DistilRoBERTa مع [`AutoModelForMaskedLM`] :

```python
>>> from transformers import AutoModelForMaskedLM

>>> model = AutoModelForMaskedLM.from_pretrained("distilbert/distilroberta-base")
```

في هذه المرحلة، لم يتبق سوى ثلاث خطوات:

1. حدد فرط معلمات التدريب الخاصة بك في [`TrainingArguments`]. المعلمة المطلوبة الوحيدة هي `output_dir` التي تحدد مكان حفظ نموذجك. ستقوم بدفع هذا النموذج إلى المركز عن طريق تعيين `push_to_hub=True` (يجب أن تكون قد سجلت الدخول إلى Hugging Face لتحميل نموذجك).
2. قم بتمرير فرط معلمات التدريب إلى [`Trainer`] جنبًا إلى جنب مع النموذج ومجموعات البيانات ومجمع البيانات.
3. استدعاء [`~ Trainer.train`] لضبط نموذجك بشكل دقيق.

```python
>>> training_args = TrainingArguments(
...     output_dir="my_awesome_eli5_mlm_model"،
...     eval_strategy="epoch"،
...     learning_rate=2e-5،
...     num_train_epochs=3،
...     weight_decay=0.01،
...     push_to_hub=True،
... )

>>> trainer = Trainer(
...     model=model،
...     args=training_args،
...     train_dataset=lm_dataset ["train"]،
...     eval_dataset=lm_dataset ["test"]،
...     data_collator=data_collator،
... )

>>> trainer.train()
```

بمجرد اكتمال التدريب، استخدم طريقة [`~ transformers.Trainer.evaluate`] لتقييم نموذجك والحصول على حيرته:

```python
>>> import math

>>> eval_results = trainer.evaluate()
>>> print (f "Perplexity: {math.exp (eval_results ['eval_loss']): .2f}")
حيرة: 8.76
```

ثم شارك نموذجك في المركز باستخدام طريقة [`~ transformers.Trainer.push_to_hub`] حتى يتمكن الجميع من استخدام نموذجك:

```python
>>> trainer.push_to_hub()
```

إذا لم تكن معتادًا على ضبط نموذج باستخدام Keras، فراجع البرنامج التعليمي الأساسي [هنا] (../training # train-a-tensorflow-model-with-keras) !

لضبط نموذج دقيق في TensorFlow، ابدأ بإعداد دالة محسن ومعدل تعلم وجدول زمني وبعض فرط معلمات التدريب:

```python
>>> from transformers import create_optimizer، AdamWeightDecay

>>> optimizer = AdamWeightDecay (learning_rate=2e-5، weight_decay_rate=0.01)
```

بعد ذلك، يمكنك تحميل DistilRoBERTa مع [`TFAutoModelForMaskedLM`] :

```python
>>> from transformers import TFAutoModelForMaskedLM

>>> model = TFAutoModelForMaskedLM.from_pretrained("distilbert/distilroberta-base")
```

قم بتحويل مجموعات البيانات الخاصة بك إلى تنسيق `tf.data.Dataset` باستخدام [`~ transformers.TFPreTrainedModel.prepare_tf_dataset`] :

```python
>>> tf_train_set = model.prepare_tf_dataset (
...     lm_dataset ["train"]،
...     shuffle=True،
...     batch_size=16،
...     collate_fn=data_collator،
... )

>>> tf_test_set = model.prepare_tf_dataset (
...     lm_dataset ["test"]،
...     shuffle=False،
...     batch_size=16،
...     collate_fn=data_collator،
... )
```

قم بتكوين النموذج للتدريب مع [`compile`] (https://keras.io/api/models/model_training_apis/#compile-method). لاحظ أن جميع نماذج Transformers بها دالة خسارة ذات صلة بالمهمة بشكل افتراضي، لذا فأنت لست بحاجة إلى تحديد واحدة ما لم ترغب في ذلك:

```python
>>> import tensorflow as tf

>>> model.compile (optimizer=optimizer) # لا توجد حجة الخسارة!
```

يمكن القيام بذلك عن طريق تحديد المكان الذي ستدفع فيه نموذجك ومعالجتك في [`~ transformers.PushToHubCallback`] :

```python
>>> from transformers.keras_callbacks import PushToHubCallback

>>> callback = PushToHubCallback (
...     output_dir="my_awesome_eli5_mlm_model"،
...     tokenizer=tokenizer،
... )
```

أخيرًا، أنت على استعداد لبدء تدريب نموذجك! استدعاء [`fit`] (https://keras.io/api/models/model_training_apis/#fit-method) مع مجموعات البيانات التدريبية والتحقق من صحتها، وعدد العصور، واستدعاء الإرجاع الخاص بك لضبط النموذج بدقة:

```python
>>> model.fit (x=tf_train_set، validation_data=tf_test_set، epochs=3، callbacks = [callback])
```

بمجرد اكتمال التدريب، يتم تحميل نموذجك تلقائيًا إلى المركز حتى يتمكن الجميع من استخدامه!

للحصول على مثال أكثر تعمقًا حول كيفية ضبط نموذج بدقة لوضع نمذجة اللغة المقنعة، راجع الدفتر المقابل
[دفتر ملاحظات PyTorch] (https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling.ipynb)
أو [دفتر TensorFlow] (https://colab.research.google.com/github/huggingface/notebooks/blob/main/examples/language_modeling-tf.ipynb).

## الاستنتاج

رائع، الآن بعد أن ضبطت نموذجًا بدقة، يمكنك استخدامه للاستنتاج!

فكر في بعض النصوص التي تريدها النموذج لملء الفراغ بها، واستخدم الرمز `<mask>` للإشارة إلى الفراغ:

```python
>>> text = "The Milky Way is a <mask> galaxy."
```

أبسط طريقة لتجربة نموذجك الدقيق للاستنتاج هي استخدامه في [`pipeline`]. قم بتنفيذ مثيل `pipeline` لملء القناع باستخدام نموذجك، ومرر نصك إليه. إذا كنت ترغب في ذلك، يمكنك استخدام معلمة `top_k` لتحديد عدد التنبؤات التي يجب إرجاعها:

```python
>>> from transformers import pipeline

>>> mask_filler = pipeline ("fill-mask"، "username/my_awesome_eli5_mlm_model")
>>> mask_filler (text، top_k=3)
[{'score': 0.5150994658470154،
'token': 21300،
'token_str': ' لولبية'،
'sequence': 'The Milky Way is a spiral galaxy.'}،
{'score': 0.07087188959121704،
'token': 2232،
'token_str': ' ضخمة'،
'sequence': 'The Milky Way is a massive galaxy.'}،
{'score': 0.06434620916843414،
'token': 650،
'token_str': ' صغيرة'،
'sequence': 'The Milky Way is a small galaxy. '}]
```

قم برمزية النص وإرجاع `input_ids` كرموز PyTorch. ستحتاج أيضًا إلى تحديد موضع الرمز `<mask>` :

```python
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained ("username/my_awesome_eli5_mlm_model")
>>> inputs = tokenizer (text، return_tensors="pt")
>>> mask_token_index = torch.where (inputs ["input_ids"] == tokenizer.mask_token_id) [1]
```

مرر المدخلات إلى النموذج وأعد `logits` للرمز المقنع:

```python
>>> from transformers import AutoModelForMaskedLM

>>> model = AutoModelForMaskedLM.from_pretrained ("username/my_awesome_eli5_mlm_model")
>>> logits = model (** inputs). logits
>>> mask_token_logits = logits [0، mask_token_index،:]
```

ثم قم بإرجاع الرموز الثلاثة المقنعة ذات الاحتمالية الأعلى وطباعتها:

```python
>>> top_3_tokens = torch.topk (mask_token_logits، 3، dim=1). indices [0]. tolist ()

>>> for token in top_3_tokens:
... print (النص.replace (معالج. رمز القناع، معالج. فك التشفير ([الرمز])))
طريق اللبن هو مجرة لولبية.
طريق اللبن هو مجرة ضخمة.
طريق اللبن هو مجرة صغيرة.
```

قم برمزية النص وإرجاع `input_ids` كرموز TensorFlow. ستحتاج أيضًا إلى تحديد موضع الرمز `<mask>` :

```python
>>> from transformers import AutoTokenizer

>>> tokenizer = AutoTokenizer.from_pretrained ("username/my_awesome_eli5_mlm_model")
>>> inputs = tokenizer (text، return_tensors="tf")
>>> mask_token_index = tf.where (inputs ["input_ids"] == tokenizer.mask_token_id) [0، 1]
```

مرر المدخلات إلى النموذج وأعد `logits` للرمز المقنع:

```python
>>> from transformers import TFAutoModelForMaskedLM

>>> model = TFAutoModelForMaskedLM.from_pretrained ("username/my_awesome_eli5_mlm_model")
>>> logits = model (** inputs). logits
>>> mask_token_logits = logits [0، mask_token_index،:]
```

ثم قم بإرجاع الرموز الثلاثة المقنعة ذات الاحتمالية الأعلى وطباعتها:

```python
>>> top_3_tokens = tf.math.top_k (mask_token_logits، 3). indices. numpy ()

>>> for token in top_3_tokens:
... print (النص.replace (معالج. رمز القناع، معالج. فك التشفير ([الرمز])))
طريق اللبن هو مجرة لولبية.
طريق اللبن هو مجرة ضخمة.
طريق اللبن هو مجرة صغيرة.
```