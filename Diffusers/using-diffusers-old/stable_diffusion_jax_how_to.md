# JAX/Flax

يدعم 🤗 Diffusers استخدام Flax للحصول على أداء استدلال فائق السرعة على وحدات معالجة TensorFlow (TPUs) من Google، مثل تلك المتوفرة في Colab أو Kaggle أو Google Cloud Platform. يُظهر هذا الدليل كيفية تشغيل الاستدلال باستخدام Stable Diffusion مع JAX/Flax.

قبل البدء، تأكد من تثبيت المكتبات اللازمة:

```py
# قم بإلغاء التعليق لتثبيت المكتبات اللازمة في Colab
#! pip install -q jax==0.3.25 jaxlib==0.3.25 flax transformers ftfy
#! pip install -q diffusers
```

يجب أيضًا التأكد من استخدام خلفية وحدة معالجة TensorFlow (TPU). في حين أن JAX لا يعمل حصريًا على وحدات معالجة TensorFlow (TPUs)، فستحصل على أفضل أداء على وحدة معالجة TensorFlow (TPU) لأن كل خادم يحتوي على 8 مسرعات وحدة معالجة TensorFlow (TPU) تعمل بالتوازي.

إذا كنت تشغل هذا الدليل في Colab، فحدد "Runtime" في القائمة أعلاه، ثم حدد خيار "Change runtime type"، ثم حدد "TPU" ضمن إعداد "Hardware accelerator". قم باستيراد JAX والتحقق بسرعة مما إذا كنت تستخدم وحدة معالجة TensorFlow (TPU):

```python
import jax
import jax.tools.colab_tpu
jax.tools.colab_tpu.setup_tpu()

num_devices = jax.device_count()
device_type = jax.devices()[0].device_kind

print(f"Found {num_devices} JAX devices of type {device_type}.")
assert (
    "TPU" in device_type,
    "Available device is not a TPU, please select TPU from Runtime > Change runtime type > Hardware accelerator",
)
# تم العثور على 8 أجهزة JAX من نوع Cloud TPU.
```

الآن، يمكنك استيراد بقية التبعيات التي ستحتاجها:

```python
import jax.numpy as jnp
from jax import pmap
from flax.jax_utils import replicate
from flax.training.common_utils import shard

from diffusers import FlaxStableDiffusionPipeline
```

## تحميل نموذج

Flax عبارة عن إطار عمل وظيفي، لذلك فإن النماذج لا تحتوي على حالة وتتم تخزين المعلمات خارجها. يقوم تحميل خط أنابيب Flax المُدرب مسبقًا بإرجاع كل من خط الأنابيب ووزن النموذج (أو المعلمات). في هذا الدليل، ستستخدم `bfloat16`، وهو نوع نصف عائم أكثر كفاءة تدعمه وحدات معالجة TensorFlow (TPUs) (يمكنك أيضًا استخدام `float32` للحصول على دقة كاملة إذا أردت).

```python
dtype = jnp.bfloat16
pipeline, params = FlaxStableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    revision="bf16",
    dtype=dtype,
)
```

## الاستدلال

عادةً ما تحتوي وحدات معالجة TensorFlow (TPUs) على 8 أجهزة تعمل بالتوازي، لذا دعنا نستخدم نفس المحفز لكل جهاز. هذا يعني أنه يمكنك إجراء الاستدلال على 8 أجهزة في نفس الوقت، مع قيام كل جهاز بتوليد صورة واحدة. ونتيجة لذلك، ستحصل على 8 صور في نفس الوقت الذي يستغرقه الشريحة لتوليد صورة واحدة!

<Tip>
تعرف على المزيد من التفاصيل في قسم [كيف تعمل الموازاة؟](#how-does-parallelization-work).
</Tip>

بعد استنساخ المحفز، احصل على معرفات النص المعلم بالرموز عن طريق استدعاء دالة `prepare_inputs` على خط الأنابيب. يتم تعيين طول النص المعلم بالرموز إلى 77 رمزًا كما هو مطلوب من خلال تكوين نموذج CLIP النصي الأساسي.

```python
prompt = "لقطة سينمائية لفيلم عن Morgan Freeman يلعب دور Jimi Hendrix، صورة شخصية، عدسة 40 مم، عمق ضحل للحقل، لقطة قريبة، إضاءة منقسمة، سينمائي"
prompt = [prompt] * jax.device_count()
prompt_ids = pipeline.prepare_inputs(prompt)
prompt_ids.shape
# (8، 77)
```

يجب تكرار معلمات النموذج والمدخلات عبر 8 أجهزة متوازية. يتم تكرار قاموس المعلمات باستخدام [`flax.jax_utils.replicate`](https://flax.readthedocs.io/en/latest/api_reference/flax.jax_utils.html#flax.jax_utils.replicate) الذي يقوم بفحص القاموس وتغيير شكل الأوزان بحيث يتم تكرارها 8 مرات. يتم تكرار المصفوفات باستخدام `shard`.

```python
# المعلمات
p_params = replicate(params)

# المصفوفات
prompt_ids = shard(prompt_ids)
prompt_ids.shape
# (8، 1، 77)
```

يعني هذا الشكل أن كل جهاز من الأجهزة الثمانية يستقبل كمدخل مصفوفة `jnp` ذات شكل `(1، 77)`، حيث `1` هو حجم الدفعة لكل جهاز. في وحدات معالجة TensorFlow (TPUs) ذات الذاكرة الكافية، يمكن أن يكون حجم الدفعة أكبر من `1` إذا كنت تريد توليد صور متعددة (لكل شريحة) في نفس الوقت.

بعد ذلك، قم بإنشاء مولد أرقام عشوائية لتمريره إلى دالة التوليد. هذه هي الإجراءات القياسية في Flax، والتي تتعامل مع الأرقام العشوائية بجدية وصرامة. من المتوقع أن تتلقى جميع الدالات التي تتعامل مع الأرقام العشوائية مولدًا لضمان إمكانية إعادة الإنتاج، حتى عند التدريب عبر أجهزة موزعة متعددة.

تستخدم دالة المساعدة أدناه بذرة لتهيئة مولد رقم عشوائي. طالما أنك تستخدم نفس البذرة، فستحصل على نفس النتائج بالضبط. لا تتردد في استخدام بذور مختلفة عند استكشاف النتائج لاحقًا في الدليل.

```python
def create_key(seed=0):
    return jax.random.PRNGKey(seed)
```

يتم تقسيم دالة المساعدة، أو `rng`، 8 مرات بحيث يتلقى كل جهاز مولدًا مختلفًا وينشئ صورة مختلفة.

```python
rng = create_key(0)
rng = jax.random.split(rng, jax.device_count())
```

للاستفادة من السرعة المحسنة لـ JAX على وحدة معالجة TensorFlow (TPU)، قم بتمرير `jit=True` إلى خط الأنابيب لتجميع رمز JAX إلى تمثيل فعال وضمان تشغيل النموذج بالتوازي عبر الأجهزة الثمانية.

<Tip warning={true}>
يجب التأكد من أن جميع مدخلاتك لها نفس الشكل في الاستدعاءات اللاحقة، وإلا فسيتعين على JAX إعادة تجميع الرمز، وهو أبطأ.
</Tip>

تستغرق عملية الاستدلال الأولى وقتًا أطول لأنها تحتاج إلى تجميع الرمز، ولكن الاستدعاءات اللاحقة (حتى مع مدخلات مختلفة) تكون أسرع بكثير. على سبيل المثال، استغرق الأمر أكثر من دقيقة لتجميع على وحدة معالجة TensorFlow (TPU) v2-8، ولكن بعد ذلك يستغرق حوالي **7 ثوانٍ** في عملية استدلال مستقبلية!

```py
%%time
images = pipeline(prompt_ids, p_params, rng, jit=True)[0]

# أوقات وحدة المعالجة المركزية: المستخدم 56.2 ثانية، النظام: 42.5 ثانية، الإجمالي: 1 دقيقة 38 ثانية
# الوقت الفعلي: 1 دقيقة 29 ثانية
```

يكون للشكل المصفوفة المُرجعة شكل `(8، 1، 512، 512، 3)` والذي يجب إعادة تشكيله لإزالة البعد الثاني والحصول على 8 صور بحجم `512 × 512 × 3`. بعد ذلك، يمكنك استخدام دالة [`~utils.numpy_to_pil`] لتحويل المصفوفات إلى صور.

```python
from diffusers.utils import make_image_grid

images = images.reshape((images.shape[0] * images.shape[1],) + images.shape[-3:])
images = pipeline.numpy_to_pil(images)
make_image_grid(images, rows=2, cols=4)
```

![img](https://huggingface.co/datasets/YiYiXu/test-doc-assets/resolve/main/stable_diffusion_jax_how_to_cell_38_output_0.jpeg)

## استخدام محفزات مختلفة

أنت لست مضطرًا بالضرورة إلى استخدام نفس المحفز على جميع الأجهزة. على سبيل المثال، لتوليد 8 محفزات مختلفة:

```python
prompts = [
    "لابرادور على طريقة هوكوساي"،
    "رسم سنجاب يتزلج في نيويورك"،
    "HAL-9000 على طريقة فان جوخ"،
    "تايمز سكوير تحت الماء، مع الأسماك ودلفين يسبحان حولها"،
    "لوحة جدارية رومانية قديمة تظهر رجلاً يعمل على جهاز كمبيوتر محمول"،
    "صورة مقربة لشابة سوداء على خلفية حضرية، بجودة عالية، بوكيه"،
    "كرسي على شكل أفوكادو"،
    "مهرج رائد فضاء في الفضاء، مع الأرض في الخلفية"،
]

prompt_ids = pipeline.prepare_inputs(prompts)
prompt_ids = shard(prompt_ids)

images = pipeline(prompt_ids, p_params, rng, jit=True).images
images = images.reshape((images.shape[0] * images.shape[1],) + images.shape[-3:])
images = pipeline.numpy_to_pil(images)

make_image_grid(images, 2, 4)
```

![img](https://huggingface.co/datasets/YiYiXu/test-doc-assets/resolve/main/stable_diffusion_jax_how_to_cell_43_output_0.jpeg)
## كيف تعمل الموازاة؟

تقوم خطوة معالجة البيانات في 🤗 Diffusers تلقائيًا بتجميع النموذج وتشغيله بشكل متوازٍ على جميع الأجهزة المتاحة. دعنا نلقي نظرة فاحصة على كيفية عمل هذه العملية.

يمكن إجراء الموازاة في JAX بعدة طرق. أسهلها هو استخدام دالة [`jax.pmap`](https://jax.readthedocs.io/en/latest/_autosummary/jax.pmap.html) لتحقيق موازاة برنامج واحد وبيانات متعددة (SPMD). وهذا يعني تشغيل عدة نسخ من نفس الكود، لكل منها مدخلات بيانات مختلفة. وهناك طرق أكثر تقدمًا ممكنة، ويمكنك الاطلاع على وثائق JAX [documentation](https://jax.readthedocs.io/en/latest/index.html) لاستكشاف هذا الموضوع بمزيد من التفصيل إذا كنت مهتمًا!

تقوم `jax.pmap` بعملين:

1. تجميع (أو "jit") الكود، وهو مشابه لـ `jax.jit()`. لا يحدث هذا عند استدعاء `pmap`، ولكن فقط في المرة الأولى التي يتم فيها استدعاء الدالة المُعالجة.
2. التأكد من أن الكود المجمع يعمل بشكل متوازٍ على جميع الأجهزة المتاحة.

وللتوضيح، قم باستدعاء `pmap` على طريقة `_generate` في خط أنابيب المعالجة (هذه طريقة خاصة تولد الصور وقد يتم إعادة تسميتها أو إزالتها في الإصدارات المستقبلية من 🤗 Diffusers):

```python
p_generate = pmap(pipeline._generate)
```

بعد استدعاء `pmap`، ستقوم الدالة المُعدة `p_generate` بما يلي:

1. إنشاء نسخة من الدالة الأساسية، `pipeline._generate`، على كل جهاز.
2. إرسال جزء مختلف من وسائط الإدخال إلى كل جهاز (هذا هو سبب ضرورة استدعاء دالة *shard*). في هذه الحالة، يتم تقسيم المصفوفة `prompt_ids` ذات الشكل `(8, 1, 77, 768)` إلى 8 أجزاء، وتتلقى كل نسخة من `_generate` إدخالًا بشكل `(1, 77, 768)`.

أهم شيء يجب الانتباه إليه هنا هو حجم الدفعة (1 في هذا المثال)، وأبعاد الإدخال التي لها معنى بالنسبة لرمزك. لا يتعين عليك تغيير أي شيء آخر لجعل الكود يعمل بشكل متوازٍ.

يستغرق الاستدعاء الأول لخط الأنابيب وقتًا أطول، ولكن الاستدعاءات اللاحقة تكون أسرع بكثير. وتُستخدم دالة `block_until_ready` لقياس وقت الاستنتاج بشكل صحيح لأن JAX يستخدم التوزيع غير المتزامن ويعيد التحكم إلى حلقة Python في أقرب وقت ممكن. لا تحتاج إلى استخدام ذلك في كودك؛ يحدث الحظر تلقائيًا عندما تريد استخدام نتيجة حساب لم يتم تحقيقه بعد.

```py
%%time
images = p_generate(prompt_ids, p_params, rng)
images = images.block_until_ready()

# CPU times: user 1min 15s, sys: 18.2 s, total: 1min 34s
# Wall time: 1min 15s
```

تحقق من أبعاد الصورة للتأكد من أنها صحيحة:

```python
images.shape
# (8, 1, 512, 512, 3)
```

## المصادر

لمعرفة المزيد عن كيفية عمل JAX مع Stable Diffusion، قد تكون مهتمًا بقراءة ما يلي:

* [تسريع استنتاج Stable Diffusion XL باستخدام JAX على Cloud TPU v5e](https://hf.co/blog/sdxl_jax)