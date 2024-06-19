# التدريب التعزيزي مع DDPO

يمكنك ضبط نموذج Stable Diffusion الدقيق على دالة المكافأة عبر التعلم التعزيزي باستخدام مكتبة 🤗 TRL و 🤗 Diffusers. يتم ذلك باستخدام خوارزمية تحسين سياسة الانتشار لإزالة التشويش (DDPO) التي قدمها Black et al. في [تدريب نماذج الانتشار باستخدام التعلم التعزيزي](https://arxiv.org/abs/2305.13301)، والتي تم تنفيذها في 🤗 TRL مع [`~trl.DDPOTrainer`].

لمزيد من المعلومات، راجع مرجع API [`~trl.DDPOTrainer`] ومنشور المدونة [ضبط نماذج Stable Diffusion الدقيقة باستخدام DDPO عبر TRL](https://huggingface.co/blog/trl-ddpo).