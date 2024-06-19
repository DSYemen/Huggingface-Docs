# تكامل XLA لنماذج TensorFlow

تعد Accelerated Linear Algebra، التي يطلق عليها XLA، مترجمًا لتسريع وقت التشغيل لنماذج TensorFlow. من [الوثائق الرسمية](https://www.tensorflow.org/xla):

XLA (Accelerated Linear Algebra) هو مترجم خاص بالمجال للجبر الخطي يمكنه تسريع نماذج TensorFlow دون أي تغييرات محتملة في كود المصدر.

إن استخدام XLA في TensorFlow أمر بسيط - فهو يأتي مضمنًا داخل مكتبة "تنسورفلو"، ويمكن تشغيله باستخدام وسيط "jit_compile" في أي دالة لإنشاء الرسم البياني مثل ["tf.function"](https://www.tensorflow.org/guide/intro_to_graphs). عند استخدام أساليب Keras مثل "fit()" و"predict()"، يمكنك تمكين XLA ببساطة عن طريق تمرير وسيط "jit_compile" إلى "model.compile()". ومع ذلك، لا تقتصر XLA على هذه الأساليب - يمكن أيضًا استخدامها لتسريع أي دالة "tf.function" عشوائية.

أعيد كتابة العديد من أساليب TensorFlow في مكتبة 🤗 Transformers لتكون متوافقة مع XLA، بما في ذلك توليد النصوص لنماذج مثل [GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2) و[T5](https://huggingface.co/docs/transformers/model_doc/t5) و[OPT](https://huggingface.co/docs/transformers/model_doc/opt)، بالإضافة إلى معالجة الكلام لنماذج مثل [Whisper](https://huggingface.co/docs/transformers/model_doc/whisper).

في حين أن مقدار التسريع يعتمد تمامًا على النموذج، فقد لاحظنا تسريعًا يبلغ حوالي 100 ضعفًا لنماذج توليد النصوص TensorFlow داخل مكتبة 🤗 Transformers. ستشرح هذه الوثيقة كيفية استخدام XLA لهذه النماذج للحصول على أقصى قدر من الأداء. كما سنقدم روابط لموارد إضافية إذا كنت مهتمًا بمعرفة المزيد حول المعايير القياسية وفلسفة التصميم الخاصة بنا وراء تكامل XLA.

## تشغيل وظائف TensorFlow باستخدام XLA

لنأخذ في الاعتبار النموذج التالي في TensorFlow:

```py
import tensorflow as tf

model = tf.keras.Sequential(
[tf.keras.layers.Dense(10, input_shape=(10,), activation="relu"), tf.keras.layers.Dense(5, activation="softmax")]
)
```

يقبل النموذج أعلاه إدخالات ذات بعد `(10, )`. يمكننا استخدام النموذج لتشغيل تمرير للأمام مثل ما يلي:

```py
# Generate random inputs for the model.
batch_size = 16
input_vector_dim = 10
random_inputs = tf.random.normal((batch_size, input_vector_dim))

# Run a forward pass.
_ = model(random_inputs)
```

لتشغيل التمرير للأمام باستخدام دالة مجمعة بواسطة XLA، سنحتاج إلى القيام بما يلي:

```py
xla_fn = tf.function(model, jit_compile=True)
_ = xla_fn(random_inputs)
```

تُستخدم دالة "call()" الافتراضية للنموذج لتجميع رسم بياني لـ XLA. ولكن إذا كان هناك أي دالة أخرى للنموذج تريد تجميعها في XLA، فيمكنك ذلك باستخدام ما يلي:

```py
my_xla_fn = tf.function(model.my_xla_fn, jit_compile=True)
```

## تشغيل نموذج توليد نص TensorFlow باستخدام XLA من 🤗 Transformers

لتمكين التوليد المعجل بواسطة XLA داخل مكتبة 🤗 Transformers، تحتاج إلى تثبيت إصدار حديث من "transformers". يمكنك تثبيته عن طريق تشغيل ما يلي:

```bash
pip install transformers --upgrade
```

بعد ذلك، يمكنك تشغيل الكود التالي:

```py
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM

# Will error if the minimal version of Transformers is not installed.
from transformers.utils import check_min_version

check_min_version("4.21.0")


tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", padding_side="left", pad_token="</s>")
model = TFAutoModelForCausalLM.from_pretrained("openai-community/gpt2")
input_string = ["TensorFlow is"]

# One line to create an XLA generation function
xla_generate = tf.function(model.generate, jit_compile=True)

tokenized_input = tokenizer(input_string, return_tensors="tf")
generated_tokens = xla_generate(**tokenized_input, num_beams=2)

decoded_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
print(f"Generated -- {decoded_text}")
# Generated -- TensorFlow is an open-source, open-source, distributed-source application # framework for the
```

كما تلاحظ، فإن تمكين XLA على "generate()" هو مجرد سطر واحد من الكود. لا يزال باقي الكود دون تغيير. ومع ذلك، هناك بعض الأشياء التي يجب مراعاتها في مقتطف الكود أعلاه والتي تخص XLA تحديدًا. يجب أن تكون على دراية بتلك الأشياء لتحقيق التسريعات التي يمكن أن توفرها XLA. نناقش هذه الأمور في القسم التالي.

## الأشياء التي يجب مراعاتها

عندما تقوم بتنفيذ دالة ممكّنة من XLA (مثل "xla_generate()" أعلاه) للمرة الأولى، فسوف تحاول داخليًا استنتاج رسم الحساب، وهو ما يستغرق وقتًا طويلاً. تُعرف هذه العملية باسم ["tracing"](https://www.tensorflow.org/guide/intro_to_graphs#when_is_a_function_tracing).

قد تلاحظ أن وقت التوليد ليس سريعًا. لن تحتاج الاستدعاءات المتتالية لـ "xla_generate()" (أو أي دالة أخرى ممكّنة من XLA) إلى استنتاج رسم الحساب، بشرط أن تتبع الإدخالات إلى الدالة نفس الشكل الذي تم بناء رسم الحساب به في البداية. في حين أن هذا ليس مشكلة بالنسبة للطرائق ذات أشكال الإدخال الثابتة (مثل الصور)، يجب الانتباه إذا كنت تعمل مع طرائق ذات أشكال إدخال متغيرة (مثل النص).

لضمان عمل "xla_generate()" دائمًا بأشكال إدخال متطابقة، يمكنك تحديد وسيطات "padding" عند استدعاء "tokenizer".

```py
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", padding_side="left", pad_token="</s>")
model = TFAutoModelForCausalLM.from_pretrained("openai-community/gpt2")
input_string = ["TensorFlow is"]

xla_generate = tf.function(model.generate, jit_compile=True)

# Here, we call the tokenizer with padding options.
tokenized_input = tokenizer(input_string, pad_to_multiple_of=8, padding=True, return_tensors="tf")

generated_tokens = xla_generate(**tokenized_input, num_beams=2)
decoded_text = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
print(f"Generated -- {decoded_text}")
```

بهذه الطريقة، يمكنك التأكد من أن الإدخالات إلى "xla_generate()" ستتلقى دائمًا إدخالات ذات الشكل الذي تم تتبعها به، مما يؤدي إلى تسريع وقت التوليد. يمكنك التحقق من ذلك باستخدام الكود التالي:

```py
import time
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2", padding_side="left", pad_token="</s>")
model = TFAutoModelForCausalLM.from_pretrained("openai-community/gpt2")

xla_generate = tf.function(model.generate, jit_compile=True)

for input_string in ["TensorFlow is", "TensorFlow is a", "TFLite is a"]:
tokenized_input = tokenizer(input_string, pad_to_multiple_of=8, padding=True, return_tensors="tf")
start = time.time_ns()
generated_tokens = xla_generate(**tokenized_input, num_beams=2)
end = time.time_ns()
print(f"Execution time -- {(end - start) / 1e6:.1f} ms\n")
```

على وحدة معالجة الرسوميات (GPU) من نوع Tesla T4، يمكنك توقع المخرجات مثل ما يلي:

```bash
Execution time -- 30819.6 ms

Execution time -- 79.0 ms

Execution time -- 78.9 ms
```

تكون الاستدعاء الأول لـ "xla_generate()" يستغرق وقتًا طويلاً بسبب التتبع، ولكن الاستدعاءات المتتالية أسرع بكثير. ضع في اعتبارك أن أي تغيير في خيارات التوليد في أي نقطة سيؤدي إلى إعادة التتبع، مما يؤدي إلى بطء وقت التوليد.

لم نغط جميع خيارات توليد النصوص التي توفرها مكتبة 🤗 Transformers في هذه الوثيقة. نشجعك على قراءة الوثائق للحصول على حالات استخدام متقدمة.

## موارد إضافية

نتركك هنا ببعض الموارد الإضافية إذا كنت ترغب في التعمق أكثر في XLA في 🤗 Transformers وبشكل عام.

* [يوفر دفتر الملاحظات هذا من Colab](https://colab.research.google.com/github/huggingface/blog/blob/main/notebooks/91_tf_xla_generate.ipynb) عرضًا توضيحيًا تفاعليًا إذا كنت ترغب في العبث بنماذج التوليد المتوافقة مع XLA (مثل [T5](https://huggingface.co/docs/transformers/model_doc/t5)) ونماذج الترميز فك الترميز (مثل [GPT2](https://huggingface.co/docs/transformers/model_doc/gpt2)).
* [تقدم هذه التدوينة في المدونة](https://huggingface.co/blog/tf-xla-generate) نظرة عامة على معايير المقارنة للنماذج المتوافقة مع XLA بالإضافة إلى مقدمة سهلة الاستخدام لـ XLA في TensorFlow.
* [تناقش هذه التدوينة في المدونة](https://blog.tensorflow.org/2022/11/how-hugging-face-improved-text-generation-performance-with-xla.html) فلسفة التصميم الخاصة بنا وراء إضافة دعم XLA إلى نماذج TensorFlow في 🤗 Transformers.
* تدوينات موصى بها لمزيد من التعلم حول XLA ورسوميات TensorFlow بشكل عام:
* [XLA: مترجم محسن لتعلم الآلة](https://www.tensorflow.org/xla)
* [مقدمة إلى الرسوميات وtf.function](https://www.tensorflow.org/guide/intro_to_graphs)
* [أداء أفضل مع tf.function](https://www.tensorflow.org/guide/function)