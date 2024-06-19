# توليد الصور غير المشروطة

ينشئ التوليد غير المشروط للصور صوراً تبدو كعينة عشوائية من بيانات التدريب التي تم تدريب النموذج عليها، لأن عملية إزالة التشويش لا تسترشد بأي سياق إضافي مثل النص أو الصورة.

للبدء، استخدم [`DiffusionPipeline`] لتحميل نقطة التحقق [anton-l/ddpm-butterflies-128] (https://huggingface.co/anton-l/ddpm-butterflies-128) لتوليد صور الفراشات. يقوم [`DiffusionPipeline`] بتنزيل جميع مكونات النموذج المطلوبة لتوليد صورة وتخزينها مؤقتًا.

```py
من diffusers استورد DiffusionPipeline

generator = DiffusionPipeline.from_pretrained("anton-l/ddpm-butterflies-128").to("cuda")
image = generator().images[0]
image
```

<Tip>

هل تريد توليد صور لشيء آخر؟ الق نظرة على دليل التدريب [guide](../training/unconditional_training) لمعرفة كيفية تدريب نموذج لتوليد صورك الخاصة.

</Tip>

صورة الإخراج هي كائن [`PIL.Image`] (https://pillow.readthedocs.io/en/stable/reference/Image.html?highlight=image#the-image-class) يمكن حفظه:

```py
image.save("generated_image.png")
```

يمكنك أيضًا تجربة المعلمة `num_inference_steps`، التي تتحكم في عدد خطوات إزالة التشويش. عادةً ما تنتج خطوات إزالة التشويش الأكثر جودة صورًا أعلى، ولكنها ستستغرق وقتًا أطول لتوليد. لا تتردد في اللعب بهذا المعلمة لمعرفة تأثيره على جودة الصورة.

```py
image = generator(num_inference_steps=100).images[0]
image
```

جرب المساحة أدناه لتوليد صورة فراشة!

<iframe
src="https://stevhliu-unconditional-image-generation.hf.space"
frameborder="0"
width="850"
height="500"
></iframe>