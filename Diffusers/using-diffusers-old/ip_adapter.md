# IP-Adapter

IP-Adapter هو محول موجه للصور يمكن توصيله بنماذج الانتشار لتمكين ميزة موجه الصور دون إجراء أي تغييرات على النموذج الأساسي. علاوة على ذلك، يمكن إعادة استخدام هذا المحول مع النماذج الأخرى التي تم ضبطها الدقيق من نفس النموذج الأساسي، ويمكن دمجه مع المحولات الأخرى مثل ControlNet. والفكرة الرئيسية وراء IP-Adapter هي آلية "الانتباه المتقاطع المنفصل" التي تضيف طبقة انتباه متقاطع منفصلة لميزات الصور بدلاً من استخدام نفس طبقة الانتباه المتقاطع لكل من ميزات النص والصورة. يسمح هذا للنموذج بتعلم المزيد من الميزات المحددة للصور.

> [!TIP]
> تعرف على كيفية تحميل محول IP في دليل [تحميل المحولات](../using-diffusers/loading_adapters)، وتأكد من الاطلاع على قسم [IP-Adapter Plus](../using-diffusers/loading_adapters#ip-adapter-plus) الذي يتطلب تحميل مشفر الصور يدويًا.

سيوضح هذا الدليل كيفية استخدام IP-Adapter لمختلف المهام والاستخدامات.
## المهام العامة

دعونا نلقي نظرة على كيفية استخدام قدرات IP-Adapter في التوجيه الصوري مع [`StableDiffusionXLPipeline`] لمهام مثل text-to-image و image-to-image و inpainting. ونشجعك أيضًا على تجربة خطوط أنابيب أخرى مثل Stable Diffusion و LCM-LoRA و ControlNet و T2I-Adapter أو AnimateDiff!

في جميع الأمثلة التالية، سترى طريقة [`~loaders.IPAdapterMixin.set_ip_adapter_scale`]. تتحكم هذه الطريقة في مقدار النص أو الصورة الشرطية التي يتم تطبيقها على النموذج. تعني قيمة `1.0` أن النموذج مشروط فقط على موجه الصورة. يؤدي خفض هذه القيمة إلى تشجيع النموذج على إنتاج صور أكثر تنوعًا، ولكن قد لا تكون متوافقة مع موجه الصورة. عادةً ما يحقق القيمة `0.5` توازنًا جيدًا بين نوعي المطالبات وينتج نتائج جيدة.

> [!TIP]
> في الأمثلة أدناه، جرّب إضافة `low_cpu_mem_usage=True` إلى طريقة [`~loaders.IPAdapterMixin.load_ip_adapter`] لتسريع وقت التحميل.

<hfoptions id="tasks">

<hfoption id="Text-to-image">

يمكن أن يكون صياغة المطالبة النصية الدقيقة لتوليد الصورة التي تريدها أمرًا صعبًا لأنها قد لا تعبر دائمًا عما تريد التعبير عنه. يساعد إضافة صورة بجانب موجه النص النموذج على فهم ما يجب عليه توليده ويمكن أن يؤدي إلى نتائج أكثر دقة.

قم بتحميل نموذج Stable Diffusion XL (SDXL) وأدخل IP-Adapter في النموذج باستخدام طريقة [`~loaders.IPAdapterMixin.load_ip_adapter`]. استخدم معلمة `subfolder` لتحميل أوزان نموذج SDXL.

```py
from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
pipeline.set_ip_adapter_scale(0.6)
```

قم بإنشاء مطالبة نصية وقم بتحميل مطالبة صورة قبل تمريرها إلى خط الأنابيب لتوليد صورة.

```py
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_diner.png")
generator = torch.Generator(device="cpu").manual_seed(0)
images = pipeline(
prompt="a polar bear sitting in a chair drinking a milkshake",
ip_adapter_image=image,
negative_prompt="deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality",
num_inference_steps=100,
generator=generator,
).images
images[0]
```

<div class="flex flex-row gap-4">
<div class="flex-1">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_diner.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">صورة IP-Adapter</figcaption>
</div>
<div class="flex-1">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_diner_2.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة المولدة</figcaption>
</div>
</div>

</hfoption>

<hfoption id="Image-to-image">

يمكن أيضًا أن يساعد IP-Adapter في image-to-image من خلال توجيه النموذج لتوليد صورة تشبه الصورة الأصلية وصورة المطالبة.

قم بتحميل نموذج Stable Diffusion XL (SDXL) وأدخل IP-Adapter في النموذج باستخدام طريقة [`~loaders.IPAdapterMixin.load_ip_adapter`]. استخدم معلمة `subfolder` لتحميل أوزان نموذج SDXL.

```py
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image
import torch

pipeline = AutoPipelineForImage2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
pipeline.set_ip_adapter_scale(0.6)
```

مرر الصورة الأصلية وصورة مطالبة IP-Adapter إلى خط الأنابيب لتوليد صورة. تقديم مطالبة نصية إلى خط الأنابيب أمر اختياري، ولكن في هذا المثال، يتم استخدام مطالبة نصية لتحسين جودة الصورة.

```py
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_bear_1.png")
ip_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_bear_2.png")

generator = torch.Generator(device="cpu").manual_seed(4)
images = pipeline(
prompt="best quality, high quality",
image=image,
ip_adapter_image=ip_image,
generator=generator,
strength=0.6,
).images
images[0]
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_bear_1.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة الأصلية</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_bear_2.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">صورة IP-Adapter</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_bear_3.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة المولدة</figcaption>
</div>
</div>

</hfoption>

<hfoption id="Inpainting">

IP-Adapter مفيد أيضًا للرتوش لأن مطالبة الصورة تسمح لك بأن تكون أكثر تحديدًا بشأن ما تريد توليده.

قم بتحميل نموذج Stable Diffusion XL (SDXL) وأدخل IP-Adapter في النموذج باستخدام طريقة [`~loaders.IPAdapterMixin.load_ip_adapter`]. استخدم معلمة `subfolder` لتحميل أوزان نموذج SDXL.

```py
from diffusers import AutoPipelineForInpainting
from diffusers.utils import load_image
import torch

pipeline = AutoPipelineForInpainting.from_pretrained("diffusers/stable-diffusion-xl-1.0-inpainting-0.1", torch_dtype=torch.float16).to("cuda")
pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")
pipeline.set_ip_adapter_scale(0.6)
```

مرر مطالبة، والصورة الأصلية، وصورة القناع، وصورة مطالبة IP-Adapter إلى خط الأنابيب لتوليد صورة.

```py
mask_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_mask.png")
image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_bear_1.png")
ip_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_gummy.png")

generator = torch.Generator(device="cpu").manual_seed(4)
images = pipeline(
prompt="a cute gummy bear waving",
image=image,
mask_image=mask_image,
ip_adapter_image=ip_image,
generator=generator,
num_inference_steps=100,
).images
images[0]
```

<div class="flex gap-4">
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_bear_1.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة الأصلية</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_gummy.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">صورة IP-Adapter</figcaption>
</div>
<div>
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_inpaint.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">الصورة المولدة</figcaption>
</div>
</div>

</hfoption>

<hfoption id="Video">

يمكن أيضًا أن يساعدك IP-Adapter في إنشاء مقاطع فيديو أكثر توافقًا مع مطالبة النص الخاصة بك. على سبيل المثال، دعنا نحمل [AnimateDiff](../api/pipelines/animatediff) مع محول الحركة الخاص به وأدخل IP-Adapter في النموذج باستخدام طريقة [`~loaders.IPAdapterMixin.load_ip_adapter`].

> [!WARNING]
> إذا كنت تخطط لتحميل النموذج إلى وحدة المعالجة المركزية، فتأكد من تشغيله بعد تحميل IP-Adapter. عند استدعاء [`~DiffusionPipeline.enable_model_cpu_offload`] قبل تحميل IP-Adapter، فإنه يقوم بتحميل وحدة ترميز الصور إلى وحدة المعالجة المركزية وسيعيد خطأ عند محاولة تشغيل خط الأنابيب.

```py
import torch
from diffusers import AnimateDiffPipeline, DDIMScheduler, MotionAdapter
from diffusers.utils import export_to_gif
from diffusers.utils import load_image

adapter = MotionAdapter.from_pretrained("guoyww/animatediff-motion-adapter-v1-5-2", torch_dtype=torch.float16)
pipeline = AnimateDiffPipeline.from_pretrained("emilianJR/epiCRealism", motion_adapter=adapter, torch_dtype=torch.float16)
scheduler = DDIMScheduler.from_pretrained(
"emilianJR/epiCRealism",
subfolder="scheduler",
clip_sample=False,
timestep_spacing="linspace",
beta_schedule="linear",
steps_offset=1,
)
pipeline.scheduler = scheduler
pipeline.enable_vae_slicing()

pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
pipeline.enable_model_cpu_offload()
```

مرر مطالبة وصورة مطالبة إلى خط الأنابيب لتوليد مقطع فيديو قصير.

```py
ip_adapter_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_inpaint.png")

output = pipeline(
prompt="A cute gummy bear waving",
negative_prompt="bad quality, worse quality, low resolution",
ip_adapter_image=ip_adapter_image,
num_frames=16,
guidance_scale=7.5,
num_inference_steps=50,
generator=torch.Generator(device="cpu").manual_seed(0),
)
frames = output.frames[0]
export_to_gif(frames, "gummy_bear.gif")
```

<div class="flex flex-row gap-4">
<div class="flex-1">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_adapter_inpaint.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">صورة IP-Adapter</figcaption>
</div>
<div class="flex-1">
<img class="rounded-xl" src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/gummy_bear.gif"/>
<figcaption class="mt-SUhface.co/documentation-images/resolve/main/diffusers/gummy_bear.gif</figcaption>
</div>
</div>

</hfoption>

</hfoptions>
## تكوين المعلمات
هناك بعض معلمات محول بروتوكول الإنترنت (IP-Adapter) التي من المفيد معرفتها والتي يمكن أن تساعدك في مهام إنشاء الصور. يمكن أن تجعل هذه المعلمات سير عملك أكثر كفاءة أو تمنحك مزيدًا من التحكم في إنشاء الصور.

### تضمين الصور
توفر خطوط أنابيب محول بروتوكول الإنترنت المعلمة `ip_adapter_image_embeds` لقبول تضمينات الصور المحسوبة مسبقًا. وهذا مفيد بشكل خاص في السيناريوهات التي تحتاج فيها إلى تشغيل خط أنابيب محول بروتوكول الإنترنت عدة مرات لأن لديك أكثر من صورة واحدة. على سبيل المثال، [محولات بروتوكول إنترنت المتعددة](#multi-ip-adapter) هي حالة استخدام محددة حيث توفر العديد من صور الأسلوب لإنشاء صورة محددة بأسلوب محدد. لن يكون كفؤًا تحميل وترميز العديد من الصور في كل مرة تستخدم فيها خط الأنابيب. بدلاً من ذلك، يمكنك حساب تضمينات الصور مسبقًا وحفظها على القرص (والذي يمكن أن يوفر الكثير من المساحة إذا كنت تستخدم صور عالية الجودة) وتحميلها عند الحاجة.

> [!TIP]
> تمنحك هذه المعلمة أيضًا المرونة لتحميل التضمينات من مصادر أخرى. على سبيل المثال، تتوافق تضمينات صور ComfyUI لمحولات بروتوكول الإنترنت مع أجهزة النشر التفاضلي (Diffusers) ويجب أن تعمل خارج الصندوق!

استدعاء طريقة [`~StableDiffusionPipeline.prepare_ip_adapter_image_embeds`] لترميز وتوليد تضمينات الصور. بعد ذلك، يمكنك حفظها على القرص باستخدام `torch.save`.

> [!TIP]
> إذا كنت تستخدم محول بروتوكول الإنترنت مع `ip_adapter_image_embedding` بدلاً من `ip_adapter_image`، فيمكنك تعيين `load_ip_adapter(image_encoder_folder=None,...)` لأنك لا تحتاج إلى تحميل برنامج ترميز لتوليد تضمينات الصور.

```py
image_embeds = pipeline.prepare_ip_adapter_image_embeds(
ip_adapter_image=image,
ip_adapter_image_embeds=None,
device="cuda",
num_images_per_prompt=1,
do_classifier_free_guidance=True,
)

torch.save(image_embeds, "image_embeds.ipadpt")
```

الآن قم بتحميل تضمينات الصور عن طريق تمريرها إلى معلمة `ip_adapter_image_embeds`.

```py
image_embeds = torch.load("image_embeds.ipadpt")
images = pipeline(
prompt="a polar bear sitting in a chair drinking a milkshake",
ip_adapter_image_embeds=image_embeds,
negative_prompt="deformed, ugly, wrong proportion, low res, bad anatomy, worst quality, low quality",
num_inference_steps=100,
generator=generator,
).images
```

### قناع محول بروتوكول الإنترنت
تحدد الأقنعة الثنائية الجزء من صورة الإخراج الذي يجب تعيينه لمحول بروتوكول الإنترنت. هذا مفيد لتكوين أكثر من صورة واحدة لمحول بروتوكول الإنترنت. يجب توفير قناع ثنائي لكل صورة من صور محول بروتوكول الإنترنت المدخلة.

للبدء، قم بمعالجة صور محول بروتوكول الإنترنت المدخلة باستخدام [`~image_processor.IPAdapterMaskProcessor.preprocess()`] لتوليد أقنعتهم. للحصول على نتائج مثالية، قم بتوفير الارتفاع والعرض للإخراج إلى [`~image_processor.IPAdapterMaskProcessor.preprocess()`]. يضمن ذلك تمدد الأقنعة ذات نسب العرض إلى الارتفاع المختلفة بشكل مناسب. إذا كانت الأقنعة المدخلة تتطابق بالفعل مع نسبة العرض إلى الارتفاع للصورة المولدة، فلا يلزمك تعيين الارتفاع والعرض.

```py
from diffusers.image_processor import IPAdapterMaskProcessor

mask1 = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_mask_mask1.png")
mask2 = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_mask_mask2.png")

output_height = 1024
output_width = 1024

processor = IPAdapterMaskProcessor()
masks = processor.preprocess([mask1, mask2], height=output_height, width=output_width)
```

عندما يكون هناك أكثر من صورة واحدة لمحول بروتوكول الإنترنت، قم بتحميلها كقائمة وتوفير قائمة مقاييس محول بروتوكول الإنترنت. تتوافق كل صورة من صور محول بروتوكول الإنترنت المدخلة هنا مع واحدة من الأقنعة المولدة أعلاه.

```py
pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name=["ip-adapter-plus-face_sdxl_vit-h.safetensors"])
pipeline.set_ip_adapter_scale([[0.7, 0.7]]) # مقياس واحد لكل زوج من الصور والقناع

face_image1 = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_mask_girl1.png")
face_image2 = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_mask_girl2.png")

ip_images = [[face_image1, face_image2]]

masks = [masks.reshape(1, masks.shape[0], masks.shape[2], masks.shape[3])]
```

الآن قم بتمرير الأقنعة المعالجة مسبقًا إلى `cross_attention_kwargs` في مكالمة خط الأنابيب.

```py
generator = torch.Generator(device="cpu").manual_seed(0)
num_images = 1

image = pipeline(
prompt="2 girls",
ip_adapter_image=ip_images,
negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
num_inference_steps=20,
num_images_per_prompt=num_images,
generator=generator,
cross_attention_kwargs={"ip_adapter_masks": masks}
).images[0]
image
```

## حالات استخدام محددة
تعد إمكانية مطالبة الصور في محول بروتوكول الإنترنت وتوافقه مع المحولات النموذجية الأخرى تجعلها أداة متعددة الاستخدامات لمجموعة متنوعة من حالات الاستخدام. يغطي هذا القسم بعض التطبيقات الأكثر شيوعًا لمحول بروتوكول الإنترنت، ولا يمكننا الانتظار لمعرفة ما ستتوصل إليه!
### نموذج الوجه

إن توليد وجوه دقيقة أمر صعب لأنها معقدة ودقيقة. يدعم Diffusers نقطتي تفتيش IP-Adapter المدربتين خصيصًا لتوليد الوجوه من مستودع [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter):

- يستخدم [ip-adapter-full-face_sd15.safetensors](https://huggingface.co/h94/IP-Adapter/blob/main/models/ip-adapter-full-face_sd15.safetensors) صور الوجوه المقصوصة والخلفيات المزالة.

- يستخدم [ip-adapter-plus-face_sd15.safetensors](https://huggingface.co/h94/IP-Adapter/blob/main/models/ip-adapter-plus-face_sd15.safetensors) تضمين الرقع وهو مشروط بصور الوجوه المقصوصة.

بالإضافة إلى ذلك، يدعم Diffusers جميع نقاط تفتيش IP-Adapter المدربة باستخدام تضمينات الوجه المستخرجة بواسطة نماذج الوجه `insightface`. النماذج المدعومة هي من مستودع [h94/IP-Adapter-FaceID](https://huggingface.co/h94/IP-Adapter-FaceID).

بالنسبة لنماذج الوجه، استخدم نقطة تفتيش [h94/IP-Adapter](https://huggingface.co/h94/IP-Adapter). يوصى أيضًا باستخدام [`DDIMScheduler`] أو [`EulerDiscreteScheduler`] لنماذج الوجه.

<معلومات>

للاطلاع على أمثلة حول كيفية استخدام نماذج الوجه، راجع قسم "نماذج الوجه" في دليل المستخدم.

### نماذج Face ID متعددة المحولات

لاستخدام نماذج Face ID متعددة المحولات، قم أولاً باستخراج تضمينات الوجه باستخدام `insightface`. ثم قم بتمرير قائمة المتوترات إلى الأنبوب كـ `ip_adapter_image_embeds`.

```py
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from diffusers.utils import load_image
from insightface.app import FaceAnalysis

pipeline = StableDiffusionPipeline.from_pretrained(
"runwayml/stable-diffusion-v1-5",
torch_dtype=torch.float16,
).to("cuda")
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
pipeline.load_ip_adapter("h94/IP-Adapter-FaceID", subfolder=None, weight_name="ip-adapter-faceid_sd15.bin", image_encoder_folder=None)
pipeline.set_ip_adapter_scale(0.6)

image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ip_mask_girl1.png")

ref_images_embeds = []
app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
faces = app.get(image)
image = torch.from_numpy(faces[0].normed_embedding)
ref_images_embeds.append(image.unsqueeze(0))
ref_images_embeds = torch.stack(ref_images_embeds, dim=0).unsqueeze(0)
neg_ref_images_embeds = torch.zeros_like(ref_images_embeds)
id_embeds = torch.cat([neg_ref_images_embeds, ref_images_embeds]).to(dtype=torch.float16, device="cuda")

generator = torch.Generator(device="cpu").manual_seed(42)

images = pipeline(
prompt="A photo of a girl",
ip_adapter_image_embeds=[id_embeds],
negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
num_inference_steps=20, num_images_per_prompt=1,
generator=generator
).images
```

تتطلب كل من نماذج IP-Adapter FaceID Plus وPlus v2 تضمينات صور CLIP. يمكنك إعداد تضمينات الوجه كما هو موضح سابقًا، ثم يمكنك استخراج تمرير تضمينات CLIP إلى طبقات الإسقاط الصورة المخفية.

```py
from insightface.utils import face_align

ref_images_embeds = []
ip_adapter_images = []
app = FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
image = cv2.cvtColor(np.asarray(image), cv2.COLOR_BGR2RGB)
faces = app.get(image)
ip_adapter_images.append(face_align.norm_crop(image, landmark=faces[0].kps, image_size=224))
image = torch.from_numpy(faces[0].normed_embedding)
ref_images_embeds.append(image.unsqueeze(0))
ref_images_embeds = torch.stack(ref_images_embeds, dim=0).unsqueeze(0)
neg_ref_images_embeds = torch.zeros_like(ref_images_embeds)
id_embeds = torch.cat([neg_ref_images_embeds, ref_images_embeds]).to(dtype=torch.float16, device="cuda")

clip_embeds = pipeline.prepare_ip_adapter_image_embeds(
[ip_adapter_images], None, torch.device("cuda"), num_images, True)[0]

pipeline.unet.encoder_hid_proj.image_projection_layers[0].clip_embeds = clip_embeds.to(dtype=torch.float16)
pipeline.unet.encoder_hid_proj.image_projection_layers[0].shortcut = False # True if Plus v2
```

### IP-Adapter متعدد

يمكن استخدام أكثر من محول IP واحد في نفس الوقت لتوليد صور محددة بأساليب أكثر تنوعًا. على سبيل المثال، يمكنك استخدام IP-Adapter-Face لتوليد وجوه وشخصيات متسقة، وIP-Adapter Plus لتوليد تلك الوجوه بأسلوب محدد.

> [!نصيحة]
> اقرأ قسم [IP-Adapter Plus](../using-diffusers/loading_adapters#ip-adapter-plus) لمعرفة السبب في ضرورة تحميل مشفر الصور يدويًا.

قم بتحميل مشفر الصور باستخدام [`~transformers. CLIPVisionModelWithProjection`].

```py
import torch
from diffusers import AutoPipelineForText2Image, DDIMScheduler
from transformers import CLIPVisionModelWithProjection
from diffusers.utils import load_image

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
"h94/IP-Adapter",
subfolder="models/image_encoder",
torch_dtype=torch.float16,
)
```

بعد ذلك، ستقوم بتحميل نموذج أساسي، ومخطط، ومحولات IP. يتم تمرير محولات IP لاستخدامها كقائمة إلى معلمة `weight_name`:

- يستخدم [ip-adapter-plus_sdxl_vit-h](https://huggingface.co/h94/IP-Adapter#ip-adapter-for-sdxl-10) تضمين الرقع ومشفر الصور ViT-H

- [ip-adapter-plus-face_sdxl_vit-h](https://huggingface.co/h94/IP-Adapter#ip-adapter-for-sdxl-10) له نفس البنية ولكنه مشروط بصور الوجوه المقصوصة

```py
pipeline = AutoPipelineForText2Image.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0",
torch_dtype=torch.float16,
image_encoder=image_encoder,
)
pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
pipeline.load_ip_adapter(
"h94/IP-Adapter",
subfolder="sdxl_models",
weight_name=["ip-adapter-plus_sdxl_vit-h.safetensors"، "ip-adapter-plus-face_sdxl_vit-h.safetensors"]
)
pipeline.set_ip_adapter_scale([0.7، 0.3])
pipeline.enable_model_cpu_offload()
```

قم بتحميل موجه صورة ومجلد يحتوي على صور لأسلوب معين تريد استخدامه.

```py
face_image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/women_input.png")
style_folder = "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/style_ziggy"
style_images = [load_image(f"{style_folder}/img{i}.png") for i in range(10)]
```

<div class="flex flex-row gap-4">
<div class="flex-1">
<img class="rounded-xl" src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/women_input.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">صورة محول IP للوجه</figcaption>
</div>
<div class="flex-1">
<img class="rounded-xl" src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/ip_style_grid.png"/>
<figcaption class="mt-2 text-center text-sm text-gray-500">صور نمط محول IP</figcaption>
</div>
</div>

قم بتمرير موجه صورة الصورة وصور الأسلوب كقائمة إلى معلمة `ip_adapter_image`، ثم قم بتشغيل الأنبوب!

```py
generator = torch.Generator(device="cpu").manual_seed(0)

image = pipeline(
prompt="wonderwoman"،
ip_adapter_image=[style_images, face_image]،
negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality"،
num_inference_steps=50, num_images_per_prompt=1،
generator=generator،
).images[0]
image
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/ip_multi_out.png" />
</div>
### التوليد الفوري

تعد نماذج الاتساق الكامنة (LCM) نماذج انتشار يمكنها توليد الصور في 4 خطوات فقط مقارنة بنماذج الانتشار الأخرى مثل SDXL التي تتطلب عادة عددًا أكبر من الخطوات. وهذا هو السبب في أن توليد الصور باستخدام LCM يشعر بأنه "فوري". يمكن توصيل وحدات IP-Adapter بنموذج LCM-LoRA لتوليد الصور على الفور باستخدام موجه صورة.

يجب تحميل أوزان IP-Adapter أولاً، ثم يمكنك استخدام [`~StableDiffusionPipeline.load_lora_weights`] لتحميل نمط LoRA والوزن الذي تريد تطبيقه على صورتك.

يجب تحميل أوزان IP-Adapter أولاً، ثم يمكنك استخدام الصورة التي تريد تطبيقها على صورتك.

جرِّب الاستخدام مع مقياس IP-Adapter أقل للتأثير بشكل أكبر على توليد الصور [herge_style](https://huggingface.co/sd-dreambooth-library/herge-style) checkpoint، وتذكر استخدام الرمز الخاص "herge_style" في موجهك لتشغيل التطبيق.

```py
pipeline.set_ip_adapter_scale(0.4)
```

الآن قم بتحميل صورة IP-Adapter وخريطة العمق.

```py
ip_adapter_image = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/statue.png")
depth_map = load_image("https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/depth.png")
```

مرر خريطة العمق وصورة IP-Adapter إلى الأنبوب لتوليد صورة.

```py
image = pipeline(
prompt="best quality, high quality",
image=depth_map,
ip_adapter_image=ip_adapter_image,
negative_prompt="monochrome, lowres, bad anatomy, worst quality, low quality",
num_inference_steps=50,
generator=generator,
).images[0]
image
```

### التحكم الهيكلي

للسيطرة على توليد الصور إلى درجة أكبر، يمكنك الجمع بين IP-Adapter ونموذج مثل [ControlNet](../using-diffusers/controlnet). ControlNet هو أيضًا محول يمكن إدراجه في نموذج الانتشار للسماح بالشرط على صورة تحكم إضافية. يمكن أن تكون صورة التحكم خرائط العمق، أو خرائط الحواف، أو تقديرات الوضع، والمزيد.

قم بتحميل نقطة تفتيش [`ControlNetModel`] المشروطة على خرائط العمق، وأدخلها في نموذج الانتشار، وقم بتحميل IP-Adapter.

```py
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
import torch
from diffusers.utils import load_image

controlnet_model_path = "lllyasviel/control_v11f1p_sd15_depth"
controlnet = ControlNetModel.from_pretrained(controlnet_model_path, torch_dtype=torch.float16)

pipeline = StableDiffusionControlNetPipeline.from_pretrained(
"runwayml/stable-diffusion-v1-5", controlnet=controlnet, torch_dtype=torch.float16)
pipeline.to("cuda")
pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin")
```

### التحكم في الأسلوب والتخطيط

[InstantStyle](https://arxiv.org/abs/2404.02733) هي طريقة قابلة للتشغيل المتبادل فوق IP-Adapter، والتي تفصل الأسلوب والتخطيط عن موجه الصورة للتحكم في توليد الصور. بهذه الطريقة، يمكنك إنشاء صور تتبع فقط الأسلوب أو التخطيط من موجه الصورة، مع تحسين التنوع بشكل كبير. يتم تحقيق ذلك عن طريق تنشيط وحدات IP-Adapters لأجزاء محددة فقط من النموذج.

بشكل افتراضي، يتم إدراج وحدات IP-Adapters في جميع طبقات النموذج. استخدم طريقة [`~loaders.IPAdapterMixin.set_ip_adapter_scale`] باستخدام قاموس لتعيين المقاييس إلى IP-Adapter في طبقات مختلفة.

```py
from diffusers import AutoPipelineForText2Image
from diffusers.utils import load_image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
pipeline.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.bin")

scale = {
"down": {"block_2": [0.0, 1.0]},
"up": {"block_0": [0.0, 1.0, 0.0]},
}
pipeline.set_ip_adapter_scale(scale)
```

سيؤدي هذا إلى تنشيط IP-Adapter في الطبقة الثانية في كتلة الجزء السفلي 2 والجزء العلوي 0 من النموذج. الأول هو الطبقة التي يتم فيها حقن معلومات التخطيط من قبل IP-Adapter والأخير يحقن الأسلوب. من خلال إدراج IP-Adapter في هاتين الطبقتين، يمكنك إنشاء صور تتبع كل من الأسلوب والتخطيط من موجه الصورة، ولكن مع محتوى أكثر تماشيًا مع موجه النص.

```py
style_image = load_image("https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg")

generator = torch.Generator(device="cpu").manual_seed(26)
image = pipeline(
prompt="a cat, masterpiece, best quality, high quality",
ip_adapter_image=style_image,
negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
guidance_scale=5,
num_inference_steps=30,
generator=generator,
).images[0]
image
```

على النقيض من ذلك، فإن إدراج IP-Adapter في جميع الطبقات غالبًا ما يؤدي إلى إنشاء صور تركز بشكل مفرط على موجه الصورة وتقلل من التنوع.

قم بتنشيط IP-Adapter فقط في طبقة الأسلوب ثم اتصل بالأنبوب مرة أخرى.

```py
scale = {
"up": {"block_0": [0.0, 1.0, 0.0]},
}
pipeline.set_ip_adapter_scale(scale)

generator = torch.Generator(device="cpu").manual_seed(26)
image = pipeline(
prompt="a cat, masterpiece, best quality, high quality",
ip_adapter_image=style_image,
negative_prompt="text, watermark, lowres, low quality, worst quality, deformed, glitch, low contrast, noisy, saturation, blurry",
guidance_scale=5,
num_inference_steps=30,
generator=generator,
).images[0]
image
```

لاحظ أنه لا يلزم تحديد جميع الطبقات في القاموس. لن يتم تضمين تلك الموجودة في القاموس افتراضيًا بمقياس 0 والذي يعني تعطيل IP-Adapter.