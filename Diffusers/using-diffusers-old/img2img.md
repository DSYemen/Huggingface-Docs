# التحويل من صورة إلى صورة

[[open-in-colab]]

تعد عملية التحويل من صورة إلى صورة مشابهة لـ [النص إلى صورة](conditional_image_generation)، ولكن بالإضافة إلى موجه، يمكنك أيضًا تمرير صورة أولية كنقطة بداية لعملية الانتشار. يتم ترميز الصورة الأولية إلى مساحة خفية ويتم إضافة ضوضاء إليها. بعد ذلك، يأخذ نموذج الانتشار الخفي موجه والصورة الخفية المضطربة، ويتوقع الضوضاء المضافة، ويزيل الضوضاء المتوقعة من الصورة الأولية الخفية للحصول على الصورة الخفية الجديدة. وأخيرًا، يقوم فك الترميز بترجمة الصورة الخفية الجديدة مرة أخرى إلى صورة.

مع 🤗 Diffusers، يكون الأمر سهلاً مثل 1-2-3:

1. قم بتحميل نقطة تفتيش في فئة [`AutoPipelineForImage2Image`]؛ يقوم هذا الأنبوب تلقائيًا بتعيين تحميل فئة الأنابيب الصحيحة بناءً على نقطة التفتيش:

```py
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid

pipeline = AutoPipelineForImage2Image.from_pretrained(
"kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16, use_safetensors=True
)
pipeline.enable_model_cpu_offload()
# احذف السطر التالي إذا لم يكن xFormers مثبتًا أو إذا كان لديك PyTorch 2.0 أو أعلى مثبتًا
pipeline.enable_xformers_memory_efficient_attention()
```

<Tip>
ستلاحظ طوال الدليل، أننا نستخدم [`~DiffusionPipeline.enable_model_cpu_offload`] و [`~DiffusionPipeline.enable_xformers_memory_efficient_attention`]، لتوفير الذاكرة وزيادة سرعة الاستدلال. إذا كنت تستخدم PyTorch 2.0، فلست بحاجة إلى استدعاء [`~DiffusionPipeline.enable_xformers_memory_efficient_attention`] على خط أنابيبك لأنه سيكون بالفعل باستخدام اهتمام [scaled-dot product](../optimization/torch2.0#scaled-dot-product-attention) الأصلي في PyTorch 2.0.
</Tip>

2. قم بتحميل صورة لتمريرها إلى خط الأنابيب:

```py
init_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png")
```

3. قم بتمرير موجه وصورة إلى خط الأنابيب لإنشاء صورة:

```py
prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"
image = pipeline(prompt, image=init_image).images[0]
make_image_grid([init_image, image], rows=1, cols=2)
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة الأولية</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة المولدة</figcaption>
</div>
</div>

## النماذج الشائعة

أكثر نماذج التحويل من صورة إلى صورة شيوعًا هي [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5)، و [Stable Diffusion XL (SDXL)](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)، و [Kandinsky 2.2](https://huggingface.co/kandinsky-community/kandinsky-2-2-decoder). تختلف نتائج نماذج Stable Diffusion و Kandinsky بسبب الاختلافات في بنيتها وعملية التدريب؛ يمكنك أن تتوقع بشكل عام أن ينتج SDXL صورًا ذات جودة أعلى من Stable Diffusion v1.5. دعونا نلقي نظرة سريعة على كيفية استخدام كل من هذه النماذج ومقارنة نتائجها.

### Stable Diffusion v1.5

Stable Diffusion v1.5 عبارة عن نموذج انتشار خفي تم تهيئته من نقطة تفتيش مبكرة، وتم ضبط دقته بشكل أكبر لـ 595 ألف خطوة على صور 512x512. لاستخدام هذا الأنبوب للتحويل من صورة إلى صورة، ستحتاج إلى إعداد صورة أولية لتمريرها إلى خط الأنابيب. بعد ذلك، يمكنك تمرير موجه والصورة إلى خط الأنابيب لإنشاء صورة جديدة:

```py
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image

pipeline = AutoPipelineForImage2Image.from_pretrained(
"runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipeline.enable_model_cpu_offload()
# احذف السطر التالي إذا لم يكن xFormers مثبتًا أو إذا كان لديك PyTorch 2.0 أو أعلى مثبتًا
pipeline.enable_xformers_memory_efficient_attention()

# إعداد الصورة
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
init_image = load_image(url)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# تمرير الموجه والصورة إلى خط الأنابيب
image = pipeline(prompt, image=init_image).images[0]
make_image_grid([init_image, image], rows=1, cols=2)
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة الأولية</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-sdv1.5.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة المولدة</figcaption>
</div>
</div>

### Stable Diffusion XL (SDXL)

SDXL هو إصدار أكثر قوة من نموذج Stable Diffusion. يستخدم نموذجًا أساسيًا أكبر، ونموذج تحسين إضافي لزيادة جودة إخراج النموذج الأساسي. اقرأ دليل [SDXL](sdxl) للحصول على شرح أكثر تفصيلاً حول كيفية استخدام هذا النموذج، والتقنيات الأخرى التي يستخدمها لإنتاج صور عالية الجودة.

```py
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image

pipeline = AutoPipelineForImage2Image.from_pretrained(
"stabilityai/stable-diffusion-xl-refiner-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
pipeline.enable_model_cpu_offload()
# احذف السطر التالي إذا لم يكن xFormers مثبتًا أو إذا كان لديك PyTorch 2.0 أو أعلى مثبتًا
pipeline.enable_xformers_memory_efficient_attention()

# إعداد الصورة
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-sdxl-init.png"
init_image = load_image(url)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# تمرير الموجه والصورة إلى خط الأنابيب
image = pipeline(prompt, image=init_image, strength=0.5).images[0]
make_image_grid([init_image, image], rows=1, cols=2)
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-sdxl-init.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة الأولية</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-sdxl.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة المولدة</figcaption>
</div>
</div>

### Kandinsky 2.2

يختلف نموذج Kandinsky عن نماذج Stable Diffusion لأنه يستخدم نموذجًا أوليًا للصور لإنشاء ترميزات للصور. تساعد الترميزات في إنشاء محاذاة أفضل بين النص والصور، مما يسمح لنموذج الانتشار الخفي بتوليد صور أفضل.

أبسط طريقة لاستخدام Kandinsky 2.2 هي:

```py
import torch
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import make_image_grid, load_image

pipeline = AutoPipelineForImage2Image.from_pretrained(
"kandinsky-community/kandinsky-2-2-decoder", torch_dtype=torch.float16, use_safetensors=True
)
pipeline.enable_model_cpu_offload()
# احذف السطر التالي إذا لم يكن xFormers مثبتًا أو إذا كان لديك PyTorch 2.0 أو أعلى مثبتًا
pipeline.enable_xformers_memory_efficient_attention()

# إعداد الصورة
url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"
init_image = load_image(url)

prompt = "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"

# تمرير الموجه والصورة إلى خط الأنابيب
image = pipeline(prompt, image=init_image).images[0]
make_image_grid([init_image, image], rows=1, cols=2)
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-init.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة الأولية</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/img2img-kandinsky.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة المولدة</figcaption>
</div>
</div>

## تكوين معلمات خط الأنابيب

هناك العديد من المعلمات المهمة التي يمكنك تكوينها في خط الأنابيب والتي ستؤثر على عملية إنشاء الصور وجودة الصورة. دعونا نلقي نظرة فاحصة على ما تفعله هذه المعلمات وكيف يؤثر تغييرها على الإخراج.
بالتأكيد، سأتبع تعليماتك في الترجمة.

### القوة

تعد "القوة" strength أحد أهم المعلمات التي يجب مراعاتها، وسيكون لها تأثير كبير على الصورة التي يتم إنشاؤها. فهي تحدد مدى تشابه الصورة الناتجة مع الصورة الأولية. وبعبارة أخرى:

- 📈 تعطي قيمة "قوة" أعلى model المزيد من "الإبداع" لإنشاء صورة تختلف عن الصورة الأولية؛ وتعني قيمة "قوة" strength تبلغ 1.0 تجاهل الصورة الأولية إلى حد كبير.

- 📉 تعني قيمة "قوة" أقل أن الصورة الناتجة أكثر تشابهاً مع الصورة الأولية.

هناك ارتباط بين معلمتي "القوة" strength و "عدد خطوات الاستدلال" num_inference_steps لأن "القوة" strength تحدد عدد خطوات الضجيج التي سيتم إضافتها. على سبيل المثال، إذا كانت "عدد خطوات الاستدلال" num_inference_steps تساوي 50 و "القوة" strength تساوي 0.8، فإن هذا يعني إضافة 40 (50 * 0.8) خطوة من الضجيج إلى الصورة الأولية، ثم إزالة الضجيج لـ 40 خطوة للحصول على الصورة المنشأة حديثًا.

### مقياس التوجيه

يُستخدم معلمة "مقياس التوجيه" guidance_scale للتحكم في مدى توافق الصورة المنشأة مع موجه النص text prompt. تشير قيمة "مقياس التوجيه" guidance_scale أعلى إلى أن الصورة المنشأة أكثر توافقاً مع الموجه prompt، في حين أن قيمة "مقياس التوجيه" guidance_scale أقل تعني أن الصورة المنشأة لديها مساحة أكبر للانحراف عن الموجه prompt.

يمكنك الجمع بين "مقياس التوجيه" guidance_scale و "القوة" strength لمزيد من التحكم الدقيق في مدى تعبير النموذج. على سبيل المثال، يمكنك الجمع بين "قوة" strength و "مقياس توجيه" guidance_scale عاليين للإبداع الأقصى، أو استخدام مزيج من "قوة" strength و "مقياس توجيه" guidance_scale منخفضين لإنشاء صورة تشبه الصورة الأولية ولكنها ليست مقيدة بشكل صارم بالموجه prompt.

### الموجه السلبي

يُستخدم الموجه السلبي negative prompt لتوجيه النموذج إلى عدم تضمين أشياء في صورة، ويمكن استخدامه لتحسين جودة الصورة أو تعديلها. على سبيل المثال، يمكنك تحسين جودة الصورة من خلال تضمين موجهات سلبية مثل "تفاصيل سيئة" أو "ضبابية" لتشجيع النموذج على إنشاء صورة ذات جودة أعلى. أو يمكنك تعديل صورة عن طريق تحديد الأشياء التي سيتم استبعادها من الصورة.

## خطوط أنابيب الصورة إلى الصورة المتسلسلة

هناك بعض الطرق المثيرة للاهتمام الأخرى التي يمكنك من خلالها استخدام خط أنابيب الصورة إلى الصورة image-to-image pipeline بخلاف مجرد إنشاء صورة (على الرغم من أن ذلك رائع أيضًا). يمكنك المضي قدمًا وتسلسله مع خطوط أنابيب أخرى.

### النص إلى الصورة إلى الصورة

يتيح تسلسل خط أنابيب النص إلى الصورة text-to-image وخط أنابيب الصورة إلى الصورة image-to-image إمكانية إنشاء صورة من النص واستخدام الصورة المنشأة كصورة أولية لخط أنابيب الصورة إلى الصورة image-to-image pipeline. هذا مفيد إذا كنت تريد إنشاء صورة من الصفر. على سبيل المثال، دعونا نقوم بتسلسل نموذج Stable Diffusion ونموذج Kandinsky.

ابدأ بإنشاء صورة باستخدام خط أنابيب النص إلى الصورة text-to-image pipeline:

الآن يمكنك تمرير هذه الصورة المنشأة إلى خط أنابيب الصورة إلى الصورة image-to-image pipeline:
بالتأكيد، سألتزم بالتعليمات المذكورة.

### سلسلة من الصور إلى الصور إلى الصور
يمكنك أيضًا توصيل العديد من خطوط الصور إلى الصور لإنشاء صور أكثر إثارة. يمكن أن يكون هذا مفيدًا لأداء نقل الأسلوب بشكل تكراري على صورة، أو إنشاء صور GIF قصيرة، أو استعادة الألوان إلى صورة، أو استعادة المناطق المفقودة من صورة.

ابدأ بتوليد صورة:

تُعد صور الفضاء طريقة رائعة لاستكشاف الكون من راحة منزلك. يمكن أن تساعدك على معرفة المزيد عن الأجسام السماوية والظواهر الفلكية، بالإضافة إلى تقديم لمحة عن روعة الكون. سواء كنت مهتمًا بالتصوير الفلكي كهواية أو كمجال للدراسة العلمية، يمكن أن يكون الوصول إلى الصور الفلكية عالية الجودة خطوة أولى رائعة.

### الصورة إلى المحسن إلى فائق الدقة
هناك طريقة أخرى لتوصيل خط أنابيب الصورة إلى الصورة الخاصة بك وهي استخدام برنامج المحسن والخط فائق الدقة لزيادة مستوى التفاصيل في الصورة حقًا.

ابدأ بخط أنابيب الصورة إلى الصورة:

تُعد صور الفضاء طريقة رائعة لاستكشاف الكون من راحة منزلك. يمكن أن تساعدك على معرفة المزيد عن الأجسام السماوية والظواهر الفلكية، بالإضافة إلى تقديم لمحة عن روعة الكون. سواء كنت مهتمًا بالتصوير الفلكي كهواية أو كمجال للدراسة العلمية، يمكن أن يكون الوصول إلى الصور الفلكية عالية الجودة خطوة أولى رائعة.

### التحكم في إنشاء الصور
يمكن أن يكون محاولة إنشاء صورة تبدو بالضبط كما تريد أمرًا صعبًا، وهذا هو السبب في أن تقنيات ونماذج التحكم في الإنشاء مفيدة جدًا. في حين يمكنك استخدام "سلبى برومبيت" للتحكم جزئيًا في إنشاء الصور، هناك طرق أكثر قوة مثل وزن الفاصل الزمني وشبكات التحكم.

### وزن الفاصل الزمني
يسمح وزن الفاصل الزمني بتغيير مقياس تمثيل كل مفهوم في موجه. على سبيل المثال، في موجه مثل "رائد فضاء في الغابة، لوحة ألوان باردة، ألوان خافتة، تفاصيل، 8k"، يمكنك اختيار زيادة أو تقليل تضمين "رائد الفضاء" و "الغابة". توفر مكتبة [Compel] (https://github.com/damian0815/compel) بناء جملة بسيطًا لتعديل أوزان الفواصل الزمنية وإنشاء التضمينات. يمكنك معرفة كيفية إنشاء التضمينات في دليل [وزن الفاصل الزمني]. (weighted_prompts)

لدى [AutoPipelineForImage2Image] معلمة "prompt_embeds" (و"negative_prompt_embeds" إذا كنت تستخدم موجهًا سلبيًا) حيث يمكنك تمرير التضمينات التي تحل محل معلمة "الموجه".

أرجو أن تكون الترجمة واضحة ومفهومة، وإذا كانت هناك أي أجزاء تحتاج إلى توضيح أو تعديل، يرجى إعلامي بذلك.
بالتأكيد! سأبدأ الترجمة من هنا:

### ControlNet

توفر ControlNets طريقة أكثر مرونة ودقة للتحكم في توليد الصور لأنك يمكنك استخدام صورة شرطية إضافية. يمكن أن تكون الصورة الشرطية صورة Canny أو خريطة عمق أو تجزئة صورة، وحتى الخربشات! أيًا كان نوع الصورة الشرطية التي تختارها، تقوم ControlNet بتوليد صورة تحافظ على المعلومات الموجودة فيها.

على سبيل المثال، دعونا نشترط صورة باستخدام خريطة عمق للحفاظ على المعلومات المكانية في الصورة.

لنشترط الآن صورة جديدة بناءً على خريطة العمق والصورة الأولية والنص:

"رائد فضاء في الغابة، لوحة ألوان باردة، ألوان خافتة، مفصلة، 8K"

دعونا نطبق [نمطًا](https://huggingface.co/nitrosocke/elden-ring-diffusion) جديدًا على الصورة التي تم إنشاؤها من ControlNet عن طريق ربطها بخط أنابيب من الصورة إلى الصورة:

```py
pipeline = AutoPipelineForImage2Image.from_pretrained(
"nitrosocke/elden-ring-diffusion", torch_dtype=torch.float16,
)
pipeline.enable_model_cpu_offload()
# remove the following line if xFormers is not installed or you have PyTorch 2.0 or higher installed
pipeline.enable_xformers_memory_efficient_attention()

prompt = "elden ring style astronaut in a jungle" # include the token "elden ring style" in the prompt
negative_prompt = "ugly, deformed, disfigured, poor details, bad anatomy"

image_elden_ring = pipeline(prompt, negative_prompt=negative_prompt, image=image_control_net, strength=0.45, guidance_scale=10.5).images[0]
make_image_grid([init_image, depth_image, image_control_net, image_elden_ring], rows=2, cols=2)
```

## تحسين

إن تشغيل نماذج الانتشار باهظ التكلفة وكثيف الاستخدام للحوسبة، ولكن ببعض الحيل التحسينية، من الممكن تمامًا تشغيلها على معالجات الرسوميات (GPUs) الاستهلاكية والمجانية. على سبيل المثال، يمكنك استخدام شكل أكثر كفاءة من حيث الذاكرة للاهتمام مثل اهتمام المنتج النقطي المُمَيز في PyTorch 2.0 أو xFormers (يمكنك استخدام أحدهما، ولكن لا توجد حاجة لاستخدام كليهما). يمكنك أيضًا نقل النموذج إلى وحدة معالجة الرسومات (GPU) أثناء انتظار المكونات الأخرى للخط على وحدة المعالجة المركزية (CPU).

باستخدام [`torch.compile`](../optimization/torch2.0#torchcompile)، يمكنك زيادة سرعة الاستدلال لديك عن طريق لفها حول UNet:

لمعرفة المزيد، راجع دليلي [تقليل استخدام الذاكرة](../optimization/memory) و [Torch 2.0](../optimization/torch2.0).