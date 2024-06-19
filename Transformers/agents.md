# الوكلاء والأدوات 

### ما هو الوكيل؟ 

يمكن للنظم اللغوية الكبيرة (LLMs) التي تم تدريبها على أداء نمذجة اللغة السببية التعامل مع مجموعة واسعة من المهام، ولكنها غالبًا ما تواجه صعوبات في المهام الأساسية مثل المنطق والعمليات الحسابية والبحث. وعندما يتم استدعاؤها في مجالات لا تؤدي فيها أداءً جيدًا، فإنها غالبًا ما تفشل في توليد الإجابة التي نتوقعها منها. 

يتمثل أحد النهج للتغلب على هذا القصور في إنشاء "وكيل". 

الوكيل هو نظام يستخدم LLM كمحرك، ولديه حق الوصول إلى وظائف تسمى "الأدوات". 

هذه "الأدوات" هي وظائف لأداء مهمة معينة، وتحتوي على جميع الأوصاف اللازمة للوكيل لاستخدامها بشكل صحيح. 

يمكن برمجة الوكيل للقيام بما يلي: 

- ابتكار سلسلة من الإجراءات/الأدوات وتشغيلها جميعًا في نفس الوقت، مثل [`CodeAgent`] على سبيل المثال 
- التخطيط للاجراءات/الأدوات وتنفيذها واحدة تلو الأخرى، والانتظار حتى يتم الانتهاء من كل إجراء قبل بدء التالي، مثل [`ReactJsonAgent`] على سبيل المثال 

### أنواع الوكلاء 

#### وكيل الشفرة 

يقوم هذا الوكيل بالتخطيط أولاً، ثم يقوم بتوليد شفرة Python لتنفيذ جميع إجراءاته في وقت واحد. وهو يتعامل بشكلٍ أصلي مع أنواع مختلفة من المدخلات والمخرجات لأدواته، وبالتالي فهو الخيار الموصى به للمهام متعددة الوسائط. 

#### وكلاء التفاعل 

هذا هو الوكيل الذي يتم اللجوء إليه لحل مهام الاستدلال، حيث يجعل إطار عمل ReAct (Yao et al.، 2022) من الكفاءة في التفكير بناءً على ملاحظاته السابقة. 

نقوم بتنفيذ إصدارين من ReactJsonAgent: 

- [`ReactJsonAgent`] يقوم بتوليد استدعاءات الأدوات على شكل JSON في مخرجاته. 
- [`ReactCodeAgent`] هو نوع جديد من ReactJsonAgent يقوم بتوليد استدعاءات أدواته على شكل مقاطع من الشفرة البرمجية، والتي تعمل بشكل جيد مع LLMs التي تتمتع بأداء قوي في الترميز. 

> [!TIP]
> اقرأ منشور المدونة [Open-source LLMs as LangChain Agents](https://huggingface.co/blog/open-source-llms-as-agents) لمعرفة المزيد عن وكيل ReAct. 

![إطار عمل وكيل ReAct](https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/blog/open-source-llms-as-agents/ReAct.png) 

على سبيل المثال، فيما يلي كيفية عمل وكيل ReAct في طريقه للإجابة على السؤال التالي. 

```py3
>>> agent.run(
...     "How many more blocks (also denoted as layers) in BERT base encoder than the encoder from the architecture proposed in Attention is All You Need?",
... )
=====New task=====
How many more blocks (also denoted as layers) in BERT base encoder than the encoder from the architecture proposed in Attention is All You Need?
====Agent is executing the code below:
bert_blocks = search(query="number of blocks in BERT base encoder")
print("BERT blocks:", bert_blocks)
====
Print outputs:
BERT blocks: twelve encoder blocks

====Agent is executing the code below:
attention_layer = search(query="number of layers in Attention is All You Need")
print("Attention layers:", attention_layer)
====
Print outputs:
Attention layers: Encoder: The encoder is composed of a stack of N = 6 identical layers. Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position- 2 Page 3 Figure 1: The Transformer - model architecture.

====Agent is executing the code below:
bert_blocks = 12
attention_layers = 6
diff = bert_blocks - attention_layers
print("Difference in blocks:", diff)
final_answer(diff)
====

Print outputs:
Difference in blocks: 6

Final answer: 6
``` 

### كيف يمكنني بناء وكيل؟ 

لبدء تشغيل الوكيل، تحتاج إلى هذه الحجج: 

- LLM لتشغيل الوكيل - الوكيل ليس هو LLM بالضبط، بل هو أكثر مثل برنامج يستخدم LLM كمحرك. 
- موجه النظام: ما الذي سيتم استدعاء محرك LLM به لتوليد مخرجاته 
- صندوق أدوات يختار الوكيل منه الأدوات للتنفيذ 
- محلل لاستخراج الأدوات التي سيتم استدعاؤها من مخرجات LLM والحجج التي سيتم استخدامها 

عند بدء تشغيل نظام الوكيل، يتم استخدام سمات الأداة لتوليد وصف للأداة، ثم يتم تضمينها في `system_prompt` للوكيل لإعلامه بالأدوات التي يمكنه استخدامها ولماذا. 

في البداية، يرجى تثبيت `agents` extras لتثبيت جميع التبعيات الافتراضية. 

```bash
pip install transformers[agents]
``` 

قم ببناء محرك LLM الخاص بك من خلال تعريف طريقة `llm_engine` التي تقبل قائمة من [الرسائل](./chat_templating.) وتعيد النص. يجب أن تقبل هذه الدالة القابلة للاستدعاء أيضًا وسيطة `stop` التي تشير إلى متى يجب التوقف عن التوليد. 

```python
from huggingface_hub import login, InferenceClient

login("<YOUR_HUGGINGFACEHUB_API_TOKEN>")

client = InferenceClient(model="meta-llama/Meta-Llama-3-70B-Instruct")

def llm_engine(messages, stop_sequences=["Task"]) -> str:
response = client.chat_completion(messages, stop=stop_sequences, max_tokens=1000)
answer = response.choices[0].message.content
return answer
``` 

يمكنك استخدام أي طريقة `llm_engine` طالما أنها: 

1. تتبع تنسيق [الرسائل](./chat_templating.md) كمدخلات لها (`List [Dict [str، str]]`) وتعيد `str` 
2. تتوقف عن توليد المخرجات عند تسلسل المرور في وسيطة `stop` 

أنت بحاجة أيضًا إلى وسيطة `tools` التي تقبل قائمة من `Tools`. يمكنك توفير قائمة فارغة لـ `tools`، ولكن استخدم صندوق الأدوات الافتراضي مع الحجة الاختيارية `add_base_tools=True`. 

الآن يمكنك إنشاء وكيل، مثل [`CodeAgent`]، وتشغيله. وللراحة، نقدم أيضًا فئة [`HfEngine`] التي تستخدم `huggingface_hub.InferenceClient` تحت الغطاء. 

```python
from transformers import CodeAgent, HfEngine

llm_engine = HfEngine(model="meta-llama/Meta-Llama-3-70B-Instruct")
agent = CodeAgent(tools=[], llm_engine=llm_engine, add_base_tools=True)

agent.run(
"Could you translate this sentence from French, say it out loud and return the audio.",
sentence="Où est la boulangerie la plus proche?",
)
``` 

سيكون هذا مفيدًا في حالة الطوارئ عند الحاجة إلى الخبز الفرنسي! 

يمكنك حتى ترك وسيطة `llm_engine` غير محددة، وسيتم إنشاء [`HfEngine`] افتراضيًا. 

```python
from transformers import CodeAgent

agent = CodeAgent(tools=[], add_base_tools=True)

agent.run(
"Could you translate this sentence from French, say it out loud and give me the audio.",
sentence="Où est la boulangerie la plus proche?",
)
``` 

لاحظ أننا استخدمنا وسيطة `sentence` الإضافية: يمكنك تمرير النص كوسيطات إضافية إلى النموذج. 

يمكنك أيضًا استخدام هذا للإشارة إلى مسار الملفات المحلية أو البعيدة للنموذج لاستخدامها: 

```py
from transformers import ReactCodeAgent

agent = ReactCodeAgent(tools=[], llm_engine=llm_engine, add_base_tools=True)

agent.run("Why does Mike not know many people in New York?", audio="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/recording.mp3")
``` 

تم تعريف موجه المخرجات والمحلل تلقائيًا، ولكن يمكنك فحصها بسهولة عن طريق استدعاء `system_prompt_template` على وكيلك. 

```python
print(agent.system_prompt_template)
``` 

من المهم أن تشرح بأوضح ما يمكن المهمة التي تريد تنفيذها. 

كل عملية [`~Agent.run`] مستقلة، ونظرًا لأن الوكيل مدعوم من LLM، فقد تؤدي الاختلافات الطفيفة في موجهك إلى نتائج مختلفة تمامًا. 

يمكنك أيضًا تشغيل الوكيل بشكل متتالي لمهام مختلفة: في كل مرة يتم فيها إعادة تهيئة سمات `agent.task` و`agent.logs`. 

#### تنفيذ الشفرة 

يقوم مفسر Python بتنفيذ الشفرة على مجموعة من المدخلات التي يتم تمريرها جنبًا إلى جنب مع أدواتك. 

يجب أن يكون هذا آمنًا لأن الوظائف الوحيدة التي يمكن استدعاؤها هي الأدوات التي قدمتها (خاصة إذا كانت أدوات من Hugging Face فقط) ووظيفة الطباعة، لذا فأنت مقيد بالفعل بما يمكن تنفيذه. 

كما أن مفسر Python لا يسمح بالاستيراد بشكل افتراضي خارج قائمة آمنة، لذا فإن جميع الهجمات الأكثر وضوحًا لا ينبغي أن تكون مشكلة. 

يمكنك أيضًا السماح بالاستيرادات الإضافية عن طريق تمرير الوحدات النمطية المصرح بها كقائمة من السلاسل في وسيطة `additional_authorized_imports` عند تهيئة [`ReactCodeAgent`] أو [`CodeAgent`]: 

```py
>>> from transformers import ReactCodeAgent

>>> agent = ReactCodeAgent(tools=[], additional_authorized_imports=['requests', 'bs4'])
>>>agent.run("Could you get me the title of the page at url 'https://huggingface.co/blog'?")

(...)
'Hugging Face – Blog'
``` 

سيتم إيقاف التنفيذ عند أي شفرة تحاول تنفيذ عملية غير قانونية أو إذا كان هناك خطأ عادي في Python في الشفرة التي تم إنشاؤها بواسطة الوكيل. 

> [!WARNING]
> يمكن لـ LLM توليد شفرة تعسفية سيتم تنفيذها: لا تضف أي استيرادات غير آمنة!## موجه النظام

يقوم الوكيل، أو بالأحرى LLM الذي يقود الوكيل، بتوليد مخرجات بناءً على موجه النظام. يمكن تخصيص موجه النظام وتصميمه ليناسب المهمة المقصودة. على سبيل المثال، تحقق من موجه النظام لـ [`ReactCodeAgent`] (النسخة أدناه مبسطة قليلًا).

ستعطى مهمة لحلها بأفضل ما يمكن.

لديك حق الوصول إلى الأدوات التالية:
<<tool_descriptions>>

لحل المهمة، يجب عليك التخطيط للمضي قدمًا في سلسلة من الخطوات، في دورة من تسلسلات "الفكر:" و"الرمز:" و"الملاحظة:".

في كل خطوة، في تسلسل "الفكر:"، يجب عليك أولاً شرح منطقك لحل المهمة، ثم الأدوات التي تريد استخدامها.

بعد ذلك، في تسلسل "الرمز:"، يجب عليك كتابة التعليمات البرمجية بلغة Python بسيطة. يجب أن ينتهي تسلسل التعليمات البرمجية بـ/ تسلسل "نهاية التعليمات البرمجية".

خلال كل خطوة وسيطة، يمكنك استخدام "print()" لحفظ أي معلومات مهمة ستحتاجها بعد ذلك.

سيتم بعد ذلك توفير مخرجات هذه الطباعة في حقل "الملاحظة:"، لاستخدام هذه المعلومات كإدخال للخطوة التالية.

في النهاية، يجب عليك إرجاع إجابة نهائية باستخدام أداة "final_answer".

فيما يلي بعض الأمثلة باستخدام أدوات افتراضية:
---
{أمثلة}

استخدمت الأمثلة أعلاه أدوات افتراضية قد لا تكون موجودة لديك. لديك حق الوصول فقط إلى تلك الأدوات:
<<أسماء الأدوات>>
يمكنك أيضًا إجراء الحسابات في التعليمات البرمجية Python التي تقوم بتوليدها.

قم دائمًا بتوفير تسلسل "الفكر:" و"الرمز: \ n ``` py" الذي ينتهي بتسلسل "``` <end_code>". يجب عليك توفير تسلسل "الرمز:" على الأقل للمضي قدمًا.

تذكر ألا تقوم بعدد كبير جدًا من العمليات في كتلة رمز واحدة! يجب عليك تقسيم المهمة إلى كتل رمز وسيطة.

قم بطباعة النتائج في نهاية كل خطوة لحفظ النتائج الوسيطة. ثم استخدم final_answer () لإرجاع النتيجة النهائية.

تأكد من تعريف جميع المتغيرات التي تستخدمها.

الآن ابدأ!

يتضمن موجه النظام:

- *مقدمة* تشرح كيف يجب أن يتصرف الوكيل وما هي الأدوات.
- وصف لجميع الأدوات التي يتم تحديدها بواسطة رمز <<tool_descriptions>> الذي يتم استبداله ديناميكيًا في وقت التشغيل بالأدوات التي يحددها المستخدم أو يختارها.
- يأتي وصف الأداة من سمات الأداة، الاسم والوصف والإدخالات وoutput_type، وقالب Jinja2 بسيط يمكنك تحسينه.
- تنسيق الإخراج المتوقع.

يمكنك تحسين موجه النظام، على سبيل المثال، عن طريق إضافة شرح لتنسيق الإخراج.

للحصول على أقصى قدر من المرونة، يمكنك استبدال قالب موجه النظام بالكامل عن طريق تمرير موجه مخصص كحجة إلى معلمة system_prompt.
```py
from transformers import ReactJsonAgent

from transformers.agents import PythonInterpreterTool

agent = ReactJsonAgent (tools = [PythonInterpreterTool ()]، system_prompt = "{your_custom_prompt}")
```
> [!تحذير]
> يرجى التأكد من تعريف سلسلة <<tool_descriptions>> في مكان ما في القالب حتى يكون الوكيل على دراية
الأدوات المتاحة.

## الأدوات

الأداة هي وظيفة ذرية يستخدمها الوكيل.

يمكنك على سبيل المثال التحقق من [أداة PythonInterpreterTool]: لديها اسم ووصف ووصف الإدخال ونوع الإخراج، وطريقة __call__ لأداء الإجراء.

عند تهيئة الوكيل، يتم استخدام سمات الأداة لتوليد وصف الأداة الذي يتم دمجه في موجه نظام الوكيل. يسمح هذا للوكيل بمعرفة الأدوات التي يمكنه استخدامها ولماذا.

## صندوق الأدوات الافتراضي

يأتي برنامج Transformers مع صندوق أدوات افتراضي لتمكين الوكلاء، يمكنك إضافته إلى وكيلك عند التهيئة باستخدام الحجة add_base_tools = True:

- **الإجابة على أسئلة المستند**: بالنظر إلى مستند (مثل ملف PDF) بتنسيق صورة، قم بالإجابة على سؤال حول هذا المستند ([دونات](./model_doc/donut))

- **الإجابة على أسئلة الصور**: بالنظر إلى صورة، قم بالإجابة على سؤال حول هذه الصورة ([فيلت](./model_doc/vilt))

- **تحويل الكلام إلى نص**: بالنظر إلى تسجيل صوتي لشخص يتحدث، قم بتفريغ الكلام إلى نص ([همس](./model_doc/whisper))

- **تحويل النص إلى كلام**: تحويل النص إلى كلام ([SpeechT5](./model_doc/speecht5))

- **الترجمة**: ترجمة جملة معينة من لغة المصدر إلى لغة الهدف.

- **مفسر كود بايثون**: يقوم بتشغيل كود بايثون الذي تم إنشاؤه بواسطة LLM في بيئة آمنة. لن تتم إضافة هذه الأداة إلى [ReactJsonAgent] إلا إذا كنت تستخدم add_base_tools=True، لأن الأدوات المستندة إلى التعليمات البرمجية يمكنها بالفعل تنفيذ كود Python

يمكنك استخدام أداة يدويًا عن طريق استدعاء وظيفة [load_tool] وإعطائها مهمة لأدائها.

من transformers استيراد load_tool

الأداة = load_tool ("النص إلى كلام")

الصوت = الأداة ("هذا هو أداة تحويل النص إلى كلام")

## إنشاء أداة جديدة

يمكنك إنشاء أداة مخصصة الخاصة بك لحالات الاستخدام التي لا تغطيها الأدوات الافتراضية من Hugging Face.

على سبيل المثال، دعنا ننشئ أداة تقوم بإرجاع أكثر النماذج تنزيلًا لمهمة معينة من Hub.

سنبدأ بالرمز أدناه.

من huggingface_hub استيراد list_models

المهمة = "تصنيف النص"

النموذج = next (iter (list_models (filter = task، sort = "التنزيلات"، direction = -1)))

طباعة (معرف النموذج)

يمكن تحويل هذا الكود إلى فئة ترث من فئة [أداة] الأساسية.

تحتاج الأداة المخصصة إلى:

- سمة الاسم، والتي تتوافق مع اسم الأداة نفسها. عادة ما يصف الاسم ما تفعله الأداة. نظرًا لأن الكود يعيد النموذج بمعظم التنزيلات لمهمة، فسنطلق عليه اسم "model_download_counter".

- سمة الوصف المستخدمة لملء موجه نظام الوكيل.

- سمة الإدخالات، والتي هي عبارة عن قاموس بمفاتيح "النوع" و"الوصف". يحتوي على معلومات تساعد مفسر Python على اتخاذ خيارات مستنيرة بشأن الإدخال.

- سمة output_type، والتي تحدد نوع الإخراج.

- طريقة forward التي تحتوي على كود الاستدلال الذي سيتم تنفيذه.

من transformers استيراد أداة

من huggingface_hub استيراد list_models

فئة HFModelDownloadsTool (أداة):

الاسم = "model_download_counter"

الوصف = (
"هذه هي الأداة التي تقوم بإرجاع أكثر النماذج تنزيلًا لمهمة معينة على Hugging Face Hub.

إنه يرجع اسم نقطة التفتيش.

)

الإدخالات = {
"المهمة": {
"النوع": "النص"،
"الوصف": "فئة المهمة (مثل تصنيف النص، تقدير العمق، إلخ)"
}
}

نوع الإخراج = "النص"

الأسلوب إلى الأمام (ذاتية، المهمة: ستر):

النموذج = next (iter (list_models (filter = task، sort = "التنزيلات"، direction = -1)))

إرجاع معرف النموذج

الآن بعد أن أصبحت فئة HFModelDownloadsTool المخصصة جاهزة، يمكنك حفظها في ملف باسم model_downloads.py واستيرادها للاستخدام.

من model_downloads استيراد HFModelDownloadsTool

الأداة = HFModelDownloadsTool ()

يمكنك أيضًا مشاركة أداتك المخصصة على Hub عن طريق استدعاء [~ Tool.push_to_hub] على الأداة. تأكد من أنك قمت بإنشاء مستودع لها على Hub وأنك تستخدم رمز وصول للقراءة.

الأداة. push_to_hub ("{your_username} / hf-model-downloads")

قم بتحميل الأداة باستخدام وظيفة [~ Tool.load_tool] ومررها إلى معلمة الأدوات في وكيلك.

من transformers استيراد load_tool، CodeAgent

model_download_tool = load_tool ("m-ric / hf-model-downloads")

الوكيل = CodeAgent (الأدوات = [model_download_tool]، llm_engine = llm_engine)

الوكيل. تشغيل (
"هل يمكنك إعطائي اسم النموذج الذي لديه معظم التنزيلات في مهمة" النص إلى الفيديو "على Hugging Face Hub؟"
)

ستحصل على ما يلي:

======== مهمة جديدة ========

هل يمكنك إعطائي اسم النموذج الذي لديه معظم التنزيلات في مهمة "النص إلى الفيديو" على Hugging Face Hub؟

==== الوكيل ينفذ الكود أدناه:

most_downloaded_model = model_download_counter (task = "النص إلى الفيديو")

طباعة (f "النموذج الأكثر تنزيلًا لمهمة" النص إلى الفيديو "هو {most_downloaded_model}.")

====

والنتيجة:

"النموذج الأكثر تنزيلًا لمهمة" النص إلى الفيديو "هو ByteDance/AnimateDiff-Lightning."

## إدارة صندوق أدوات الوكيل

إذا كنت قد قمت بالفعل بتهيئة وكيل، فمن غير الملائم إعادة تهيئته من الصفر باستخدام أداة تريد استخدامها. باستخدام برنامج Transformers، يمكنك إدارة صندوق أدوات الوكيل عن طريق إضافة أداة أو استبدالها.

دعنا نضيف أداة model_download_tool إلى وكيل موجود تمت تهيئته مسبقًا باستخدام صندوق الأدوات الافتراضي فقط.

من transformers استيراد CodeAgent

الوكيل = CodeAgent (الأدوات = []، llm_engine = llm_engine، add_base_tools = True)

الوكيل. toolbox.add_tool (model_download_tool)

الآن يمكننا الاستفادة من كل من الأداة الجديدة وأداة تحويل النص إلى كلام السابقة:

الوكيل. تشغيل (
"هل يمكنك قراءة اسم النموذج بصوت عالٍ والذي لديه معظم التنزيلات في مهمة" النص إلى الفيديو "على Hugging Face Hub وإرجاع الصوت؟"
)

| **الصوت**                                                                                                                                            |
|------------------------------------------------------------------------------------------------------------------------------------------------------|
| <audio controls><source src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/damo.wav" type="audio/wav"/> |

> [!تحذير]
> احترس عند إضافة أدوات إلى وكيل يعمل بالفعل بشكل جيد لأنه يمكن أن يؤدي إلى تحيز الاختيار تجاه أداتك أو اختيار أداة أخرى غير الأداة المحددة بالفعل.

استخدم طريقة agent.toolbox.update_tool () لاستبدال أداة موجودة في صندوق أدوات الوكيل.

هذا مفيد إذا كانت أداتك الجديدة بديلاً مباشرًا للأداة الموجودة لأن الوكيل يعرف بالفعل كيفية أداء تلك المهمة المحددة.

تأكد فقط من اتباع الأداة الجديدة لنفس واجهة برمجة التطبيقات مثل الأداة المستبدلة أو قم بتكييف قالب موجه النظام لضمان تحديث جميع الأمثلة التي تستخدم الأداة المستبدلة.### استخدام مجموعة من الأدوات
يمكنك الاستفادة من مجموعات الأدوات باستخدام كائن ToolCollection، مع slug لمجموعة الأدوات التي تريد استخدامها. ثم قم بتمريرها كقائمة لتهيئة الوكيل الخاص بك، وابدأ في استخدامها!

للتسريع من البداية، يتم تحميل الأدوات فقط إذا تم استدعاؤها بواسطة الوكيل.

### استخدام gradio-tools
[gradio-tools] (https://github.com/freddyaboulton/gradio-tools) هي مكتبة قوية تسمح باستخدام Hugging Face Spaces كأدوات. فهو يدعم العديد من المساحات الموجودة بالإضافة إلى مساحات مخصصة.

تدعم Transformers gradio_tools باستخدام طريقة [Tool.from_gradio]. على سبيل المثال، دعنا نستخدم [StableDiffusionPromptGeneratorTool] (https://github.com/freddyaboulton/gradio-tools/blob/main/gradio_tools/tools/prompt_generator.py) من مجموعة أدوات gradio-tools لتحسين المطالبات لإنشاء صور أفضل.

استورد وقم مثيل الأداة، ثم مررها إلى طريقة Tool.from_gradio:

الآن يمكنك استخدامه مثل أي أداة أخرى. على سبيل المثال، دعنا نحسن المطالبة "أرنب يرتدي بدلة فضاء".

يستفيد النموذج بشكل كافٍ من الأداة:

======== مهمة جديدة ========
حسن هذه المطالبة، ثم قم بإنشاء صورة لها.
تم تزويدك بهذه الحجج الأولية: {"prompt": "أرنب يرتدي بدلة فضاء"}.
==== ينفذ الوكيل التعليمات البرمجية أدناه:
improved_prompt = StableDiffusionPromptGenerator (query=prompt)
while improved_prompt == "QUEUE_FULL":
improved_prompt = StableDiffusionPromptGenerator (query=prompt)
print (f "المطالبة المحسنة هي {improved_prompt}.")
الصورة = image_generator (prompt=improved_prompt)
====

قبل إنشاء الصورة أخيرًا:

> [!تحذير]
> تتطلب أدوات gradio إدخالات ومخرجات نصية حتى عند العمل مع طرائق مختلفة مثل كائنات الصور والصوت. إدخالات ومخرجات الصور والصوت غير متوافقة حاليًا.

### استخدام أدوات LangChain
نحن نحب Langchain ونعتقد أنها تحتوي على مجموعة أدوات جذابة للغاية.

لاستيراد أداة من LangChain، استخدم طريقة from_langchain ().

هنا كيف يمكنك استخدامه لإعادة إنشاء نتيجة البحث التمهيدي باستخدام أداة بحث الويب LangChain.

من langchain.agents استيراد load_tools
من المحولات استيراد أداة، ReactCodeAgent