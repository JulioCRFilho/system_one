#!/usr/bin/env python3
"""Builds the native macOS 'System 1 HUD.app' bundle with a custom cyber-themed icon."""

import os
import shutil
import subprocess
from PIL import Image, ImageDraw, ImageFont

def main():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    app_path = os.path.join(project_dir, "System 1 HUD.app")
    icns_path = os.path.join(project_dir, "System1HUD.icns")
    user_apps_path = os.path.expanduser("~/Applications/System 1 HUD.app")

    print("[1/4] Gerando ícone de alta resolução (1024x1024)...")
    size = 1024
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    margin = 44
    r = 215

    # Brilho externo neon ciano
    for glow in range(24, 0, -2):
        alpha = int(45 * (1 - glow / 24))
        draw.rounded_rectangle(
            [margin - glow, margin - glow, size - margin + glow, size - margin + glow],
            radius=r + glow,
            fill=(56, 189, 248, alpha),
        )

    # Base escura com borda ciano
    draw.rounded_rectangle(
        [margin, margin, size - margin, size - margin],
        radius=r,
        fill=(8, 12, 22, 255),
        outline=(56, 189, 248, 255),
        width=18,
    )

    # Borda secundária interna
    draw.rounded_rectangle(
        [margin + 18, margin + 18, size - margin - 18, size - margin - 18],
        radius=r - 18,
        outline=(15, 23, 42, 255),
        width=8,
    )

    # Raio elétrico neon
    bolt = [
        (540, 150),
        (330, 520),
        (490, 520),
        (430, 830),
        (720, 450),
        (550, 450),
        (630, 150),
    ]

    for g in range(32, 0, -3):
        alpha = int(35 * (1 - g / 32))
        glow_bolt = [(x + (g if x > 512 else -g), y + (g if y > 490 else -g)) for x, y in bolt]
        draw.polygon(glow_bolt, fill=(56, 189, 248, alpha))

    draw.polygon(bolt, fill=(250, 204, 21, 255), outline=(56, 189, 248, 255))

    try:
        font = ImageFont.truetype("/System/Library/Fonts/SFNSMono.ttf", 105)
    except Exception:
        font = ImageFont.load_default()

    draw.text((512, 850), "S1 HUD", fill=(56, 189, 248, 255), anchor="ms", font=font)

    iconset_dir = os.path.join(project_dir, "tmp_hud.iconset")
    os.makedirs(iconset_dir, exist_ok=True)
    for s in [16, 32, 128, 256, 512]:
        img.resize((s, s), Image.Resampling.LANCZOS).save(f"{iconset_dir}/icon_{s}x{s}.png")
        img.resize((s * 2, s * 2), Image.Resampling.LANCZOS).save(f"{iconset_dir}/icon_{s}x{s}@2x.png")

    subprocess.run(["iconutil", "-c", "icns", iconset_dir, "-o", icns_path], check=True)
    shutil.rmtree(iconset_dir)

    print("[2/4] Compilando AppleScript nativo com osacompile...")
    if os.path.exists(app_path):
        shutil.rmtree(app_path)

    python_bin = os.path.join(project_dir, ".venv/bin/python")
    applescript_code = f'''
set projectDir to "{project_dir}"
set pythonBin to "{python_bin}"
set logFile to "/tmp/system1_hud.log"

-- 1. Verifica se o servidor já está ativo na porta 8050
set isRunning to false
try
    do shell script "nc -z 127.0.0.1 8050"
    set isRunning to true
end try

if not isRunning then
    -- Inicia o servidor em segundo plano desanexado do terminal
    do shell script "cd " & quoted form of projectDir & " && nohup " & quoted form of pythonBin & " -m system1_engine.hud > " & quoted form of logFile & " 2>&1 &"
    
    -- Aguarda o servidor responder na porta 8050 (até 6s)
    repeat 30 times
        delay 0.2
        try
            do shell script "nc -z 127.0.0.1 8050"
            exit repeat
        end try
    end repeat
end if

-- 2. Abre a interface no navegador padrão do macOS
do shell script "open http://127.0.0.1:8050"
'''

    tmp_scpt = os.path.join(project_dir, "launcher.applescript")
    with open(tmp_scpt, "w") as f:
        f.write(applescript_code)

    subprocess.run(["osacompile", "-o", app_path, tmp_scpt], check=True)
    os.remove(tmp_scpt)

    print("[3/4] Injetando metadados e ícone no bundle...")
    target_icns = os.path.join(app_path, "Contents/Resources/applet.icns")
    shutil.copy(icns_path, target_icns)

    plist_path = os.path.join(app_path, "Contents/Info.plist")
    if os.path.exists(plist_path):
        with open(plist_path, "r") as f:
            plist_data = f.read()
        plist_data = plist_data.replace(
            "<key>CFBundleName</key>\n\t<string>applet</string>",
            "<key>CFBundleName</key>\n\t<string>System 1 HUD</string>\n\t<key>CFBundleDisplayName</key>\n\t<string>System 1 HUD</string>"
        )
        with open(plist_path, "w") as f:
            f.write(plist_data)

    subprocess.run(["touch", app_path])

    print("[4/4] Copiando para ~/Applications...")
    os.makedirs(os.path.expanduser("~/Applications"), exist_ok=True)
    if os.path.exists(user_apps_path):
        shutil.rmtree(user_apps_path)
    shutil.copytree(app_path, user_apps_path)

    print("\n✅ Sucesso! O aplicativo 'System 1 HUD.app' está pronto:")
    print(f"   • No projeto: {app_path}")
    print(f"   • No sistema: {user_apps_path}")
    print("\n💡 Para adicionar ao Dock:")
    print("   Basta arrastar o arquivo 'System 1 HUD.app' diretamente para o seu Dock no macOS!")

if __name__ == "__main__":
    main()
