# Shap-E

Shap-E هو نموذج مشروط لتوليد أصول ثلاثية الأبعاد يمكن استخدامها في تطوير ألعاب الفيديو وتصميم الديكور الداخلي والهندسة المعمارية. وقد تم تدريبه على مجموعة كبيرة من البيانات لأصول ثلاثية الأبعاد، وتمت معالجتها بعد ذلك لإنتاج المزيد من وجهات النظر لكل كائن وإنتاج سحب نقطية بدقة 16K بدلاً من 4K. يتم تدريب نموذج Shap-E في خطوتين:

1. يقبل مشفر سحب النقاط والمناظر التي تم عرضها لأصل ثلاثي الأبعاد وينتج معلمات الدوال الضمنية التي تمثل الأصل.
2. يتم تدريب نموذج الانتشار على المعاملات المخفية التي ينتجها المشفر لتوليد حقول الإشعاع العصبية (NeRFs) أو شبكة ثلاثية الأبعاد منسوجة، مما يجعل من السهل عرض الأصل ثلاثي الأبعاد واستخدامه في التطبيقات اللاحقة.

سيوضح هذا الدليل كيفية استخدام Shap-E لبدء إنشاء أصول ثلاثية الأبعاد الخاصة بك!

قبل أن تبدأ، تأكد من تثبيت المكتبات التالية:

## النص إلى 3D

لتوليد صورة متحركة GIF لكائن ثلاثي الأبعاد، قم بتمرير موجه نصي إلى [`ShapEPipeline`]. يقوم الأنبوب بتوليد قائمة من إطارات الصور التي تستخدم لإنشاء الكائن ثلاثي الأبعاد.

```py
import torch
from diffusers import ShapEPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipe = ShapEPipeline.from_pretrained("openai/shap-e", torch_dtype=torch.float16, variant="fp16")
pipe = pipe.to(device)

guidance_scale = 15.0
prompt = ["A firecracker", "A birthday cupcake"]

images = pipe(
prompt,
guidance_scale=guidance_scale,
num_inference_steps=64,
frame_size=256,
).images
```

الآن استخدم وظيفة [`~utils.export_to_gif`] لتحويل قائمة إطارات الصور إلى صورة GIF للكائن ثلاثي الأبعاد.

```py
from diffusers.utils import export_to_gif

export_to_gif(images[0], "firecracker_3d.gif")
export_to_gif(images[1], "cake_3d.gif")
```

## الصورة إلى 3D

لإنشاء كائن ثلاثي الأبعاد من صورة أخرى، استخدم [`ShapEImg2ImgPipeline`]. يمكنك استخدام صورة موجودة أو إنشاء صورة جديدة تمامًا. دعونا نستخدم نموذج [Kandinsky 2.1](../api/pipelines/kandinsky) لإنشاء صورة جديدة.

```py
from diffusers import DiffusionPipeline
import torch

prior_pipeline = DiffusionPipeline.from_pretrained("kandinsky-community/kandinsky-2-1-prior", torch_dtype=torch.float16, use_safetensors=True).to("cuda")
pipeline = DiffusionPipeline.from_pretrained("kandinsky-community/kandinsky-2-1", torch_dtype=torch.float16, use_safetensors=True).to("cuda")

prompt = "A cheeseburger, white background"

image_embeds, negative_image_embeds = prior_pipeline(prompt, guidance_scale=1.0).to_tuple()
image = pipeline(
prompt,
image_embeds=image_embeds,
negative_image_embeds=negative_image_embeds,
).images[0]

image.save("burger.png")
```

مرر البرجر إلى [`ShapEImg2ImgPipeline`] لإنشاء تمثيل ثلاثي الأبعاد له.

```py
from PIL import Image
from diffusers import ShapEImg2ImgPipeline
from diffusers.utils import export_to_gif

pipe = ShapEImg2ImgPipeline.from_pretrained("openai/shap-e-img2img", torch_dtype=torch.float16, variant="fp16").to("cuda")

guidance_scale = 3.0
image = Image.open("burger.png").resize((256, 256))

images = pipe(
image,
guidance_scale=guidance_scale,
num_inference_steps=64,
frame_size=256,
).images

gif_path = export_to_gif(images[0], "burger_3d.gif")
```

## إنشاء شبكة

Shap-E هو نموذج مرن يمكنه أيضًا إنشاء مخرجات شبكية منسوجة لاستخدامها في التطبيقات اللاحقة. في هذا المثال، ستقوم بتحويل الإخراج إلى ملف `glb` لأن مكتبة مجموعات البيانات 🤗 تدعم تصور ملفات `glb` التي يمكن عرضها بواسطة [عارض مجموعة البيانات](https://huggingface.co/docs/hub/datasets-viewer#dataset-preview).

يمكنك إنشاء مخرجات شبكية لكل من [`ShapEPipeline`] و [`ShapEImg2ImgPipeline`] عن طريق تحديد معلمة `output_type` كـ `"mesh"`:

```py
import torch
from diffusers import ShapEPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

pipe = ShapEPipeline.from_pretrained("openai/shap-e", torch_dtype=torch.float16, variant="fp16")
pipe = pipe.to(device)

guidance_scale = 15.0
prompt = "A birthday cupcake"

images = pipe(prompt, guidance_scale=guidance_scale, num_inference_steps=64, frame_size=256, output_type="mesh").images
```

استخدم وظيفة [`~utils.export_to_ply`] لحفظ إخراج الشبكة كملف PLY:

<Tip>
يمكنك أيضًا حفظ إخراج الشبكة كملف OBJ باستخدام وظيفة [`~utils.export_to_obj`]. إن القدرة على حفظ إخراج الشبكة بتنسيقات مختلفة تجعلها أكثر مرونة للاستخدام في التطبيقات اللاحقة!
</Tip>

```py
from diffusers.utils import export_to_ply

ply_path = export_to_ply(images[0], "3d_cake.ply")
print(f"Saved to folder: {ply_path}")
```

بعد ذلك، يمكنك تحويل ملف PLY إلى ملف GLB باستخدام مكتبة trimesh:

```py
import trimesh

mesh = trimesh.load("3d_cake.ply")
mesh_export = mesh.export("3d_cake.glb", file_type="glb")
```

تركز مخرجات الشبكة بشكل افتراضي من منظور الرؤية السفلي، ولكن يمكنك تغيير منظور الرؤية الافتراضي عن طريق تطبيق تحويل الدوران:

```py
import trimesh
import numpy as np

mesh = trimesh.load("3d_cake.ply")
rot = trimesh.transformations.rotation_matrix(-np.pi / 2, [1, 0, 0])
mesh = mesh.apply_transform(rot)
mesh_export = mesh.export("3d_cake.glb", file_type="glb")
```

قم بتحميل ملف الشبكة إلى مستودع مجموعة البيانات الخاصة بك لعرضه باستخدام عارض مجموعة البيانات!

هل يمكنك مشاركة ما قمت بإنشائه مع المجتمع في منتديات Hugging Face؟ نحن نتطلع إلى رؤية إبداعاتك!