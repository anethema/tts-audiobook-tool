"""Local speech model setup and project settings controls."""

from __future__ import annotations

import gradio as gr

from tts_audiobook_tool.web_model_options import model_options_for, validate_model_options
from tts_audiobook_tool.web_model_setup import model_profiles, recommended_batch_size


class ModelPanel:
    def __init__(self, installed, default_model, settings, event_options):
        self.profiles = {profile.model_id: profile for profile in model_profiles()}
        value = default_model if default_model in self.profiles else next(iter(self.profiles), None)
        installed_ids = {model_id for _, model_id in installed}
        detail, available = self.describe(value, installed_ids)
        self.selected = gr.Dropdown(
            choices=self.choices(installed),
            value=value,
            label="Local speech models",
            interactive=True,
        )
        self.details = gr.Markdown(detail)
        with gr.Row():
            self.install = gr.Button("Install and select model", variant="primary", interactive=available)
            self.cancel = gr.Button("Cancel installation", variant="stop", interactive=False)
        self.setup_status = gr.Textbox(label="Installation status", interactive=False, visible=False)
        self.settings_status = gr.Textbox(label="Model settings status", interactive=False, visible=False)
        @gr.render(inputs=[self.selected, settings], concurrency_id=event_options["concurrency_id"])
        def render_options(model_id, all_settings):
            profile = self.profiles.get(model_id)
            if profile is None:
                return
            options = model_options_for(profile.catalog_id)
            if not options:
                return
            settings_for_model = all_settings.get(model_id, {})
            controls = []
            with gr.Group():
                gr.Markdown("### Synthesis settings")
                for option in options:
                    value = settings_for_model.get(option.key, option.default)
                    label = option.label
                    key = (model_id, option.key)
                    if option.choices:
                        labels = [name for name, _ in option.choices]
                        index = next(
                            (index for index, (_, choice) in enumerate(option.choices) if choice == value), 0,
                        )
                        controls.append(gr.Dropdown(
                            label=label, choices=labels, value=labels[index], type="index",
                            interactive=True, info=option.info or None, key=key, preserved_by_key=[],
                        ))
                    elif isinstance(option.default, str):
                        controls.append(gr.Textbox(
                            label=label, value=value, interactive=True, info=option.info or None,
                            key=key, preserved_by_key=[],
                        ))
                    else:
                        controls.append(gr.Number(
                            label=label, value=value, minimum=option.minimum, maximum=option.maximum,
                            precision=0 if option.integer else None, info=option.info or None,
                            interactive=True, key=key, preserved_by_key=[],
                        ))
                with gr.Row():
                    apply_settings = gr.Button("Apply model settings")
                    reset_settings = gr.Button("Restore model defaults")

            def apply_model_settings(current, *values):
                submitted = {
                    option.key: option.choices[value][1] if option.choices and value is not None else value
                    for option, value in zip(options, values, strict=True)
                }
                try:
                    overrides = validate_model_options(profile.catalog_id, submitted)
                except ValueError as exc:
                    return gr.skip(), gr.update(value=str(exc), visible=True)
                updated = {**current, model_id: {key: value for key, value in overrides.items() if key in submitted}}
                message = f"Applied settings for {profile.label}."
                return updated, gr.update(value=message, visible=True)

            def reset_model_settings(current):
                updated = {key: value for key, value in current.items() if key != model_id}
                return updated, gr.update(value=f"Restored defaults for {profile.label}.", visible=True)

            apply_settings.click(
                apply_model_settings, [settings, *controls], [settings, self.settings_status], **event_options,
            )
            reset_settings.click(reset_model_settings, [settings], [settings, self.settings_status], **event_options)

    def choices(self, installed):
        installed_ids = {model_id for _, model_id in installed}
        return [
            (f"{profile.label} ({'Installed' if model_id in installed_ids else 'Available to install'})", model_id)
            for model_id, profile in self.profiles.items()
        ]

    def describe(self, model_id, installed_ids):
        profile = self.profiles.get(model_id)
        if profile is None:
            return "", False
        detail = f"**{profile.label}** · Python {profile.python_version}"
        if model_id in installed_ids:
            detail += " · Installed"
        elif not profile.installable:
            detail += " · Browser installation requires x86-64 Linux or WSL."
        if profile.note:
            detail += "\n\n" + profile.note
        return detail, profile.installable and model_id not in installed_ids

    @staticmethod
    def recommended_batch(model_id, validate, all_settings, force_cpu=False):
        model_settings = all_settings.get(model_id, {})
        if force_cpu or model_settings.get("rolling_cont"):
            return 1
        return recommended_batch_size(model_id, validate_audio=validate, model_target=model_settings.get("target"))
