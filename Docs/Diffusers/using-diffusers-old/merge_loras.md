# دمج LoRAs

يمكن أن يكون استخدام عدة [LoRAs]((https://huggingface.co/docs/peft/conceptual_guides/adapter#low-rank-adaptation-lora)) معًا ممتعًا وإبداعيًا لتوليد شيء جديد تمامًا وفريد من نوعه. تعمل هذه الطريقة من خلال دمج أوزان LoRA متعددة لإنتاج صور تمزج بين أساليب مختلفة. يوفر Diffusers عدة طرق لدمج LoRAs اعتمادًا على *كيفية* الرغبة في دمج أوزانها، والتي يمكن أن تؤثر على جودة الصورة.

سيوضح هذا الدليل كيفية دمج LoRAs باستخدام الأساليب [`~loaders.UNet2DConditionLoadersMixin.set_adapters`] و [`~peft.LoraModel.add_weighted_adapter`]. لتحسين سرعة الاستدلال وتقليل استخدام الذاكرة لـ LoRAs المدمجة، سترى أيضًا كيفية استخدام طريقة [`~loaders.LoraLoaderMixin.fuse_lora`] لدمج أوزان LoRA مع الأوزان الأصلية للنموذج الأساسي.

بالنسبة لهذا الدليل، قم بتحميل نقطة تفتيش Stable Diffusion XL (SDXL) و [KappaNeuro/studio-ghibli-style]() و [Norod78/sdxl-chalkboarddrawing-lora]() LoRAs باستخدام طريقة [`~loaders.LoraLoaderMixin.load_lora_weights`]. ستحتاج إلى تعيين `adapter_name` لكل LoRA لدمجها لاحقًا.

```py
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights("ostris/ikea-instructions-lora-sdxl", weight_name="ikea_instructions_xl_v1_5.safetensors", adapter_name="ikea")
pipeline.load_lora_weights("lordjia/by-feng-zikai", weight_name="fengzikai_v1.0_XL.safetensors", adapter_name="feng")
```

## set_adapters

تدمج طريقة [`~loaders.UNet2DConditionLoadersMixin.set_adapters`] محولات LoRA عن طريق ضم مصفوفاتها المرجحة. استخدم اسم المحول لتحديد محولات LoRA التي تريد دمجها، وحدد معلمة `adapter_weights` للتحكم في التدرج لكل محول LoRA. على سبيل المثال، إذا كان `adapter_weights=[0.5, 0.5]`، فإن إخراج LoRA المدمج هو متوسط كلا المحولين. جرّب ضبط أوزان المحولين لمشاهدة تأثيره على الصورة المولدة!

```py
pipeline.set_adapters(["ikea", "feng"], adapter_weights=[0.7, 0.8])

generator = torch.manual_seed(0)
prompt = "A bowl of ramen shaped like a cute kawaii bear, by Feng Zikai"
image = pipeline(prompt, generator=generator, cross_attention_kwargs={"scale": 1.0}).images[0]
image
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/lora_merge_set_adapters.png"/>
</div>

## add_weighted_adapter

> [!تحذير]
> هذه طريقة تجريبية تضيف طرق PEFTs [`~peft.LoraModel.add_weighted_adapter`] إلى Diffusers لتمكين طرق الدمج الأكثر كفاءة. تحقق من هذه [القضية](https://github.com/huggingface/diffusers/issues/6892) إذا كنت مهتمًا بمعرفة المزيد عن الدافع والتصميم وراء هذا التكامل.

توفر طريقة [`~peft.LoraModel.add_weighted_adapter`] الوصول إلى طريقة الدمج الأكثر كفاءة مثل [TIES و DARE](https://huggingface.co/docs/peft/developer_guides/model_merging). لاستخدام طرق الدمج هذه، تأكد من تثبيت أحدث إصدار مستقر من Diffusers و PEFT.

```bash
pip install -U diffusers peft
```

هناك ثلاث خطوات لدمج LoRAs باستخدام طريقة [`~peft.LoraModel.add_weighted_adapter`]:

1. قم بإنشاء [`~peft.PeftModel`] من النموذج الأساسي ونقطة تفتيش LoRA.
2. قم بتحميل نموذج UNet الأساسي ومحولات LoRA.
3. قم بدمج المحولات باستخدام طريقة [`~peft.LoraModel.add_weighted_adapter`] وطريقة الدمج التي تختارها.

دعنا نتعمق في ما تنطوي عليه هذه الخطوات.

1. قم بتحميل UNet التي تتوافق مع UNet في نقطة تفتيش LoRA. في هذه الحالة، يستخدم كل من LoRAs UNet SDXL كنموذج أساسي.

```python
from diffusers import UNet2DConditionModel
import torch

unet = UNet2DConditionModel.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0",
torch_dtype=torch.float16,
use_safetensors=True,
variant="fp16",
subfolder="unet",
).to("cuda")
```

قم بتحميل خط أنابيب SDXL ومحولات LoRA، بدءًا من [ostris/ikea-instructions-lora-sdxl](https://huggingface.co/ostris/ikea-instructions-lora-sdxl) LoRA.

```python
from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0",
variant="fp16",
torch_dtype=torch.float16,
unet=unet
).to("cuda")
pipeline.load_lora_weights("ostris/ikea-instructions-lora-sdxl", weight_name="ikea_instructions_xl_v1_5.safetensors", adapter_name="ikea")
```

الآن ستقوم بإنشاء [`~peft.PeftModel`] من نقطة تفتيش LoRA المحملة عن طريق دمج UNet SDXL و UNet LoRA من خط الأنابيب.

```python
from peft import get_peft_model, LoraConfig
import copy

sdxl_unet = copy.deepcopy(unet)
ikea_peft_model = get_peft_model(
sdxl_unet,
pipeline.unet.peft_config["ikea"],
adapter_name="ikea"
)

original_state_dict = {f"base_model.model.{k}": v for k, v in pipeline.unet.state_dict().items()}
ikea_peft_model.load_state_dict(original_state_dict, strict=True)
```

> [!نصيحة]
> يمكنك اختياريًا دفع ikea_peft_model إلى Hub عن طريق استدعاء `ikea_peft_model.push_to_hub("ikea_peft_model", token=TOKEN)`.

كرر هذه العملية لإنشاء [`~peft.PeftModel`] من [lordjia/by-feng-zikai](https://huggingface.co/lordjia/by-feng-zikai) LoRA.

```python
pipeline.delete_adapters("ikea")
sdxl_unet.delete_adapters("ikea")

pipeline.load_lora_weights("lordjia/by-feng-zikai", weight_name="fengzikai_v1.0_XL.safetensors", adapter_name="feng")
pipeline.set_adapters(adapter_names="feng")

feng_peft_model = get_peft_model(
sdxl_unet,
pipeline.unet.peft_config["feng"],
adapter_name="feng"
)

original_state_dict = {f"base_model.model.{k}": v for k, v in pipe.unet.state_dict().items()}
feng_peft_model.load_state_dict(original_state_dict, strict=True)
```

2. قم بتحميل نموذج UNet الأساسي ثم قم بتحميل المحولات عليه.

```python
from peft import PeftModel

base_unet = UNet2DConditionModel.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0",
torch_dtype=torch.float16,
use_safetensors=True,
variant="fp16",
subfolder="unet",
).to("cuda")

model = PeftModel.from_pretrained(base_unet, "stevhliu/ikea_peft_model", use_safetensors=True, subfolder="ikea", adapter_name="ikea")
model.load_adapter("stevhliu/feng_peft_model", use_safetensors=True, subfolder="feng", adapter_name="feng")
```

3. قم بدمج المحولات باستخدام طريقة [`~peft.LoraModel.add_weighted_adapter`] وطريقة الدمج التي تختارها (تعرف على المزيد حول طرق الدمج الأخرى في [منشور المدونة هذا](https://huggingface.co/blog/peft_merging)). كمثال، دعنا نستخدم طريقة `"dare_linear"` لدمج LoRAs.

> [!تحذير]
> ضع في اعتبارك أن LoRAs يجب أن يكون لها نفس الرتبة ليتم دمجها!

```python
model.add_weighted_adapter(
adapters=["ikea", "feng"],
weights=[1.0, 1.0],
combination_type="dare_linear",
adapter_name="ikea-feng"
)
model.set_adapters("ikea-feng")
```

الآن يمكنك إنشاء صورة باستخدام LoRA المدمج.

```python
model = model.to(dtype=torch.float16, device="cuda")

pipeline = DiffusionPipeline.from_pretrained(
"stabilityai/stable-diffusion-xl-base-1.0", unet=model, variant="fp16", torch_dtype=torch.float16,
).to("cuda")

image = pipeline("A bowl of ramen shaped like a cute kawaii bear, by Feng Zikai", generator=torch.manual_seed(0)).images[0]
image
```

<div class="flex justify-center">
<img src="https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/ikea-feng-dare-linear.png"/>
</div>
## fuse_lora
تتطلب كل من طريقتي [`~loaders.UNet2DConditionLoadersMixin.set_adapters`] و [`~peft.LoraModel.add_weighted_adapter`] تحميل النموذج الأساسي ومهايئات LoRA بشكل منفصل، مما يتسبب في بعض النفقات العامة. تسمح طريقة [`~loaders.LoraLoaderMixin.fuse_lora`] بدمج أوزان LoRA مباشرة مع الأوزان الأصلية للنموذج الأساسي. بهذه الطريقة، فأنت لا تقوم بتحميل النموذج سوى مرة واحدة، مما يمكن أن يزيد من الاستدلال ويقلل من استخدام الذاكرة.

يمكنك استخدام PEFT لدمج/إلغاء دمج عدة مهايئات مباشرة في أوزان النموذج (كل من UNet وencoder النص) باستخدام طريقة [`~loaders.LoraLoaderMixin.fuse_lora`]` `، والتي يمكن أن تؤدي إلى تسريع الاستدلال وانخفاض استخدام VRAM.

على سبيل المثال، إذا كان لديك نموذج أساسي ومهايئات محملة ومحددة كنشطة مع أوزان المهايئ التالية:

```py
from diffusers import DiffusionPipeline
import torch

pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights("ostris/ikea-instructions-lora-sdxl", weight_name="ikea_instructions_xl_v1_5.safetensors", adapter_name="ikea")
pipeline.load_lora_weights("lordjia/by-feng-zikai", weight_name="fengzikai_v1.0_XL.safetensors", adapter_name="feng")

pipeline.set_adapters(["ikea", "feng"], adapter_weights=[0.7, 0.8])
```

قم بدمج هذه المهايئات LoRAs في UNet مع طريقة [`~loaders.LoraLoaderMixin.fuse_lora`]. يتحكم معامل `lora_scale` في مقدار المخرجات التي يتم تحجيمها بواسطة أوزان LoRA. من المهم إجراء تعديلات `lora_scale` في طريقة [`~loaders.LoraLoaderMixin.fuse_lora`] لأنها لن تعمل إذا حاولت تمرير `scale` إلى `cross_attention_kwargs` في الأنابيب.

```py
pipeline.fuse_lora(adapter_names=["ikea", "feng"], lora_scale=1.0)
```

بعد ذلك، يجب استخدام [`~loaders.LoraLoaderMixin.unload_lora_weights`] لإلغاء تحميل أوزان LoRA لأنها تم دمجها بالفعل مع النموذج الأساسي الأساسي. أخيرًا، اتصل بـ [`~DiffusionPipeline.save_pretrained`] لحفظ الأنبوب المدمج محليًا أو يمكنك استدعاء [`~DiffusionPipeline.push_to_hub`] لدفع الأنبوب المدمج إلى Hub.

```py
pipeline.unload_lora_weights()
# save locally
pipeline.save_pretrained("path/to/fused-pipeline")
# save to the Hub
pipeline.push_to_hub("fused-ikea-feng")
```

الآن يمكنك تحميل الأنبوب المدمج بسرعة واستخدامه للاستدلال دون الحاجة إلى تحميل مهايئات LoRA بشكل منفصل.

```py
pipeline = DiffusionPipeline.from_pretrained(
"username/fused-ikea-feng", torch_dtype=torch.float16,
).to("cuda")

image = pipeline("A bowl of ramen shaped like a cute kawaii bear, by Feng Zikai", generator=torch.manual_seed(0)).images[0]
image
```

يمكنك استدعاء [`~loaders.LoraLoaderMixin.unfuse_lora`] لاستعادة أوزان النموذج الأصلي (على سبيل المثال، إذا كنت تريد استخدام قيمة `lora_scale` مختلفة). ومع ذلك، فإن هذا يعمل فقط إذا قمت بدمج مهايئ LoRA واحد فقط في النموذج الأصلي. إذا قمت بدمج عدة مهايئات LoRAs، فسوف تحتاج إلى إعادة تحميل النموذج.

```py
pipeline.unfuse_lora()
```

### torch.compile
يمكن لـ [torch.compile](../optimization/torch2.0#torchcompile) تسريع خط أنابيبك بشكل أكبر، ولكن يجب دمج أوزان LoRA أولاً ثم إلغاء تحميلها. عادةً ما يتم تجميع UNet لأنه مكون مكثف حسابيًا في خط الأنابيب.

```py
from diffusers import DiffusionPipeline
import torch

# تحميل النموذج الأساسي وLoRAs
pipeline = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16).to("cuda")
pipeline.load_lora_weights("ostris/ikea-instructions-lora-sdxl", weight_name="ikea_instructions_xl_v1_5.safetensors", adapter_name="ikea")
pipeline.load_lora_weights("lordjia/by-feng-zikai", weight_name="fengzikai_v1.0_XL.safetensors", adapter_name="feng")

# تنشيط كلا المهايئين وتعيين أوزان المهايئ
pipeline.set_adapters(["ikea", "feng"], adapter_weights=[0.7, 0.8])

# دمج المهايئات وإلغاء تحميل الأوزان
pipeline.fuse_lora(adapter_names=["ikea", "feng"], lora_scale=1.0)
pipeline.unload_lora_weights()

# torch.compile
pipeline.unet.to(memory_format=torch.channels_last)
pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)

image = pipeline("A bowl of ramen shaped like a cute kawaii bear, by Feng Zikai", generator=torch.manual_seed(0)).images[0]
```

تعرف على المزيد حول torch.compile في دليل [تسريع الاستدلال لنماذج الانتشار النصية إلى الصور](../tutorials/fast_diffusion#torchcompile).

## الخطوات التالية
للحصول على تفاصيل مفاهيمية حول كيفية عمل كل طريقة دمج، راجع منشور المدونة [🤗 PEFT يرحب بطرق الدمج الجديدة](https://huggingface.co/blog/peft_merging#concatenation-cat)!