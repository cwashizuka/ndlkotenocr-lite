import logging
#logging.basicConfig(filename='debug.log', encoding='utf-8',level=logging.DEBUG)
import flet as ft
import sys
import os
import numpy as np
from PIL import Image
sys.path.append(os.path.join(".","src"))
import ocr
from tools.ndlkoten2tei import convert_tei
import xml.etree.ElementTree as ET
import time
from concurrent.futures import ThreadPoolExecutor
import time
import json
import shutil
import argparse
import yaml
import io
import glob
import pypdfium2

from reading_order.xy_cut.eval import eval_xml
from ndl_parser import convert_to_xml_string3

name = "NDLkotenOCR-Lite-GUI"
PDFTMPPATH="4ab7ecc3-53fb-b3e7-64e8-a809b5a483d2"


def main(page: ft.Page):
    page.title = "NDL古典籍OCR-Lite-GUI"
    page.window.icon=os.path.join("assets","icon.png")
    page.scroll = ft.ScrollMode.AUTO
    page.expand = True
    page.window.width = 1400
    page.window.height = 900
    inputpathlist=[]
    visualizepathlist=[]
    outputtxtlist=[]

    def create_pdf_func(outputpath:str,img:object,bboxlistobj:dict,viztxtflag:bool):
        import reportlab
        from reportlab.pdfgen import canvas
        from reportlab.lib.pagesizes import portrait
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.cidfonts import UnicodeCIDFont
        from reportlab.lib.units import mm
        from reportlab.lib.utils import ImageReader
        from reportlab.lib.colors import blue
        pdfmetrics.registerFont(UnicodeCIDFont('HeiseiMin-W3', isVertical=True))
        print((img.shape[1],img.shape[0]))
        c = canvas.Canvas(outputpath, pagesize=(img.shape[1],img.shape[0]))
        pilimg_data = io.BytesIO()
        pilimg=Image.fromarray(img)
        pilimg.save(pilimg_data, format='png')
        pilimg_data.seek(0)
        side_out = ImageReader(pilimg_data)
        c.drawImage(side_out,0,0)
        if viztxtflag:
            c.setFillColor(blue)
        else:
            c.setFillColor(blue,alpha=0.0)
        for bboxobj in bboxlistobj:
            bbox=bboxobj["boundingBox"]
            text=bboxobj["text"]
            x_center=(bbox[0][0]+bbox[2][0])//2
            y_center=img.shape[0]-bbox[0][1]
            c.setFont('HeiseiMin-W3', abs(bbox[2][0]-bbox[0][0])*3//4)
            c.drawString(x_center,y_center, text)
        c.save()

    def parts_control(flag:bool):
        file_upload_btn.disabled=flag
        directory_upload_btn.disabled=flag
        directory_output_btn.disabled=flag
        chkbx_visualize.disabled=flag
        customize_btn.disabled=flag
        preview_prev_btn.disabled=flag
        preview_next_btn.disabled=flag
        ocr_btn.disabled=flag
        # ★追加：テキスト保存・結合ボタンも制御
        save_text_btn.disabled=flag
        merge_text_btn.disabled=flag


    def ocr_button_result(e):
        progressbar.value=0
        outputpath=selected_output_path.value
        parser = argparse.ArgumentParser(description="Argument for YOLOv9 Inference using ONNXRuntime")
        parser.add_argument("--det-weights", type=str, required=False, help="Path to rtmdet onnx file", default="./src/model/rtmdet-s-1280x1280.onnx")
        parser.add_argument("--det-classes", type=str, required=False, help="Path to list of class in yaml file",default="./src/config/ndl.yaml")
        parser.add_argument("--det-score-threshold", type=float, required=False, default=0.3)
        parser.add_argument("--det-conf-threshold", type=float, required=False, default=0.3)
        parser.add_argument("--det-iou-threshold", type=float, required=False, default=0.3)
        parser.add_argument("--rec-weights", type=str, required=False, help="Path to parseq-tiny onnx file", default="./src/model/parseq-ndl-32x384-tiny-10.onnx")
        parser.add_argument("--rec-classes", type=str, required=False, help="Path to list of class in yaml file", default="./src/config/NDLmoji.yaml")
        parser.add_argument("--device", type=str, required=False, help="Device use (cpu or cude)", choices=["cpu", "cuda"], default="cpu")
        args = parser.parse_args()
        nonlocal inputpathlist,outputtxtlist,visualizepathlist,preview_index
        preview_index=0
        parts_control(True)
        page.update()
        progressmessage.value="Start"
        progressmessage.update()
        try:
            recognizer=ocr.get_recognizer(args=args)
            tatelinecnt=0
            alllinecnt=0
            allsum=len(inputpathlist)
            allstart=time.time()
            progressbar.value=0
            progressbar.update()
            outputtxtlist.clear()
            visualizepathlist.clear()
            visualizepathlist=[]
            alljsonobjlist=[]
            for idx,inputpath in enumerate(inputpathlist):
                progressmessage.value=inputpath
                progressmessage.update()
                pil_image = Image.open(inputpath).convert('RGB')
                npimg = np.array(pil_image)
                start = time.time()

                master_h,master_w=npimg.shape[:2]
                inputdivlist=[]
                imgnamelist=[]
                inputdivlist.append(npimg)
                imgnamelist.append(os.path.basename(inputpath))
                allxmlstr="<OCRDATASET>\n"
                alltextlist=[]
                resjsonarray=[]
                for img,imgname in zip(inputdivlist,imgnamelist):
                    img_h,img_w=img.shape[:2]
                    detections,classeslist=ocr.inference_on_detector(args=args,inputname=imgname,npimage=img,outputpath=outputpath,issaveimg=chkbx_visualize.value)
                    e1=time.time()
                    resultobj=[dict(),dict()]
                    resultobj[0][0]=list()
                    for i in range(16):
                        resultobj[1][i]=[]
                    for det in detections:
                        xmin,ymin,xmax,ymax=det["box"]
                        conf=det["confidence"]
                        if det["class_index"]==0:
                            resultobj[0][0].append([xmin,ymin,xmax,ymax])
                        resultobj[1][det["class_index"]].append([xmin,ymin,xmax,ymax,conf])

                    xmlstr=convert_to_xml_string3(img_w, img_h, imgname, classeslist, resultobj,score_thr = 0.3,min_bbox_size= 5,use_block_ad= False)
                    xmlstr="<OCRDATASET>"+xmlstr+"</OCRDATASET>"
                    root = ET.fromstring(xmlstr)
                    eval_xml(root, logger=None)
                    targetdflist=[]
                    with ThreadPoolExecutor(max_workers=8, thread_name_prefix="thread") as executor:
                        for lineobj in root.findall(".//LINE"):
                            xmin=int(lineobj.get("X"))
                            ymin=int(lineobj.get("Y"))
                            line_w=int(lineobj.get("WIDTH"))
                            line_h=int(lineobj.get("HEIGHT"))
                            if line_h>line_w:
                                tatelinecnt+=1
                            alllinecnt+=1
                            lineimg=img[ymin:ymin+line_h,xmin:xmin+line_w,:]
                            targetdflist.append(lineimg)
                        resultlines = executor.map(recognizer.read, targetdflist)
                        resultlines=list(resultlines)
                        alltextlist.append("\n".join(resultlines))
                        for idx,lineobj in enumerate(root.findall(".//LINE")):
                            lineobj.set("STRING",resultlines[idx])
                            xmin=int(lineobj.get("X"))
                            ymin=int(lineobj.get("Y"))
                            line_w=int(lineobj.get("WIDTH"))
                            line_h=int(lineobj.get("HEIGHT"))
                            try:
                                conf=float(lineobj.get("CONF"))
                            except:
                                conf=0
                            jsonobj={"boundingBox": [[xmin,ymin],[xmin,ymin+line_h],[xmin+line_w,ymin],[xmin+line_w,ymin+line_h]],
                                "id": idx,"isVertical": "true","text": resultlines[idx],"isTextline": "true","confidence": conf}
                            resjsonarray.append(jsonobj)
                    allxmlstr+=(ET.tostring(root.find("PAGE"), encoding='unicode')+"\n")
                    e2=time.time()
                allxmlstr+="</OCRDATASET>"
                if alllinecnt==0 or tatelinecnt/alllinecnt>0.5:
                    alltextlist=alltextlist[::-1]
                outputtxtlist.append("\n".join(alltextlist))
                alljsonobj={
                    "contents":[resjsonarray],
                    "imginfo": {
                        "img_width": img_w,
                        "img_height": img_h,
                        "img_path":inputpath,
                        "img_name":os.path.basename(inputpath)
                    }
                }
                alljsonobjlist.append(alljsonobj)
                if chkbx_xml.value:
                    with open(os.path.join(outputpath,os.path.basename(inputpath).split(".")[0]+".xml"),"w",encoding="utf-8") as wf:
                        wf.write(allxmlstr)
                if chkbx_json.value:
                    with open(os.path.join(outputpath,os.path.basename(inputpath).split(".")[0]+".json"),"w",encoding="utf-8") as wf:
                        wf.write(json.dumps(alljsonobj,ensure_ascii=False,indent=2))
                if chkbx_txt.value:
                    with open(os.path.join(outputpath,os.path.basename(inputpath).split(".")[0]+".txt"),"w",encoding="utf-8") as wtf:
                        wtf.write("\n".join(alltextlist))
                if chkbx_pdf.value:
                    create_pdf_func(os.path.join(outputpath,os.path.basename(inputpath).split(".")[0]+".pdf"),img,resjsonarray,chkbx_pdf_viztxt.value)
                if chkbx_visualize.value:
                    visualizepathlist.append(os.path.join(outputpath,"viz_"+os.path.basename(inputpath)))
                progressbar.value+=1/allsum
                preview_prev_btn.disabled=False
                preview_next_btn.disabled=False
                preview_text.value= outputtxtlist[preview_index]
                if len(visualizepathlist)>0:
                    preview_image.src = visualizepathlist[preview_index]
                else:
                    preview_image.src = inputpathlist[preview_index]
                update_dropdown()
                page.update()
            progressmessage.value="{} 画像OCR完了 / 処理時間: {:.2f} 秒".format(allsum,time.time()-allstart)
            progressmessage.update()
            if chkbx_tei.value:
                with open(os.path.join(outputpath,os.path.basename(inputpathlist[0]).split(".")[0]+"_tei.xml"),"wb") as wf:
                    allxmlstrtei=convert_tei(alljsonobjlist)
                    wf.write(allxmlstrtei)
        except Exception as e:
            progressmessage.value=e
            progressmessage.update()
        finally:
            parts_control(False)
            page.update()


    def pick_files_result(e: ft.FilePickerResultEvent):
        if e.files:
            selected_input_path.value=e.files[0].path
            nonlocal inputpathlist,outputtxtlist
            inputpathlist.clear()
            outputtxtlist.clear()
            ext=e.files[0].path.split(".")[-1]
            if ext=="pdf":
                filestem=os.path.basename(e.files[0].path)[:-4]
                progressmessage.value="pdfファイルの前処理中… {} ".format(e.files[0].path)
                parts_control(True)
                page.update()
                for p in glob.glob(os.path.join(os.getcwd(),PDFTMPPATH,"*.jpg")):
                    if os.path.isfile(p):
                        os.remove(p)
                os.makedirs(os.path.join(os.getcwd(),PDFTMPPATH), exist_ok=True)
                doc = pypdfium2.PdfDocument(selected_input_path.value)
                pdfarray = doc.render(pypdfium2.PdfBitmap.to_pil,scale=200 / 72)
                for ix,image in enumerate(list(pdfarray)):
                    outputtmppath=os.path.join(os.getcwd(),PDFTMPPATH,"{}_{:05}.jpg".format(filestem,ix))
                    inputpathlist.append(outputtmppath)
                    image=image.convert("RGB")
                    image.save(outputtmppath)
                progressmessage.value="pdfファイルの前処理の完了"
                parts_control(False)
                page.update()
            else:
                inputpathlist.append(e.files[0].path)
            if selected_output_path.value!=None:
                ocr_btn.disabled=False
        selected_input_path.update()
        page.update()

    def pick_directory_result(e: ft.FilePickerResultEvent):
        print(e.path)
        if e.path:
            selected_input_path.value = e.path
            nonlocal inputpathlist,outputtxtlist
            inputpathlist.clear()
            outputtxtlist.clear()
            cleanflag=False
            for inputname in os.listdir(e.path):
                inputpath=os.path.join(e.path,inputname)
                ext=inputpath.split(".")[-1]
                if ext.lower() in ["jpg","png","tiff","jp2","tif","jpeg","bmp"] and os.path.isfile(inputpath):
                    inputpathlist.append(inputpath)
                    if selected_output_path.value!=None:
                        ocr_btn.disabled=False
                elif ext=="pdf" and os.path.isfile(inputpath):
                    filestem=os.path.basename(inputpath)[:-4]
                    progressmessage.value="pdfファイルの前処理中… {} ".format(inputpath)
                    parts_control(True)
                    page.update()
                    if not cleanflag:
                        for p in glob.glob(os.path.join(os.getcwd(),PDFTMPPATH,"*.jpg")):
                            if os.path.isfile(p):
                                os.remove(p)
                        cleanflag=True
                    os.makedirs(os.path.join(os.getcwd(),PDFTMPPATH), exist_ok=True)
                    doc = pypdfium2.PdfDocument(inputpath)
                    pdfarray = doc.render(pypdfium2.PdfBitmap.to_pil,scale=200 / 72)
                    for ix,image in enumerate(list(pdfarray)):
                        outputtmppath=os.path.join(os.getcwd(),PDFTMPPATH,"{}_{:05}.jpg".format(filestem,ix))
                        inputpathlist.append(outputtmppath)
                        image=image.convert("RGB")
                        image.save(outputtmppath)
                    progressmessage.value="pdfファイルの前処理の完了"
                    parts_control(False)
                    page.update()
        selected_input_path.update()
        page.update()

    def pick_output_result(e: ft.FilePickerResultEvent):
        nonlocal inputpathlist
        if e.path:
            selected_output_path.value = e.path
            selected_output_path.update()
            if len(inputpathlist)>0:
                ocr_btn.disabled=False
        page.update()

    preview_index=0
    def next_image(e):
        nonlocal inputpathlist,outputtxtlist,preview_index
        if preview_index < min(len(inputpathlist) - 1,len(outputtxtlist) - 1):
            preview_index += 1
        else:
            preview_index = 0

        if len(visualizepathlist)>0:
            preview_image.src = visualizepathlist[preview_index]
        else:
            preview_image.src = inputpathlist[preview_index]
        preview_text.value=outputtxtlist[preview_index]
        page_dropdown.value = str(preview_index)
        page.update()


    def prev_image(e):
        nonlocal inputpathlist,outputtxtlist,preview_index
        if preview_index > 0:
            preview_index -= 1
        else:
            preview_index = min(len(inputpathlist) - 1,len(outputtxtlist) - 1)

        if len(visualizepathlist)>0:
            preview_image.src = visualizepathlist[preview_index]
        else:
            preview_image.src = inputpathlist[preview_index]
        preview_text.value=outputtxtlist[preview_index]
        page_dropdown.value = str(preview_index)
        preview_text.update()
        page.update()

    # ★追加：プレビュー画面の編集内容をtxt・xml・jsonに保存する
    def save_text(e):
        nonlocal inputpathlist, outputtxtlist, preview_index
        outputpath = selected_output_path.value
        if not outputpath:
            progressmessage.value = "出力先フォルダが選択されていません"
            progressmessage.update()
            return
        if len(inputpathlist) == 0 or len(outputtxtlist) == 0:
            progressmessage.value = "保存するテキストがありません"
            progressmessage.update()
            return
        # 編集内容をoutputtxtlistにも反映
        outputtxtlist[preview_index] = preview_text.value
        filestem = os.path.basename(inputpathlist[preview_index]).rsplit(".", 1)[0]
        saved = []

        # TXTに保存
        if chkbx_txt.value:
            txt_path = os.path.join(outputpath, filestem + ".txt")
            with open(txt_path, "w", encoding="utf-8") as wf:
                wf.write(preview_text.value)
            saved.append("txt")

        # XMLに保存（STRING属性を編集テキストで更新）
        if chkbx_xml.value:
            xml_path = os.path.join(outputpath, filestem + ".xml")
            if os.path.exists(xml_path):
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    new_lines = preview_text.value.split("\n")
                    line_elements = root.findall(".//LINE")
                    for i, lineobj in enumerate(line_elements):
                        if i < len(new_lines):
                            lineobj.set("STRING", new_lines[i])
                    ET.indent(root)
                    tree.write(xml_path, encoding="unicode", xml_declaration=False)
                    saved.append("xml")
                except Exception:
                    pass

        # JSONに保存（textフィールドを編集テキストで更新）
        if chkbx_json.value:
            json_path = os.path.join(outputpath, filestem + ".json")
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as rf:
                        jsonobj = json.load(rf)
                    new_lines = preview_text.value.split("\n")
                    contents = jsonobj.get("contents", [[]])[0]
                    for i, item in enumerate(contents):
                        if i < len(new_lines):
                            item["text"] = new_lines[i]
                    with open(json_path, "w", encoding="utf-8") as wf:
                        wf.write(json.dumps(jsonobj, ensure_ascii=False, indent=2))
                    saved.append("json")
                except Exception:
                    pass

        progressmessage.value = "保存しました: {} ({})".format(filestem, "・".join(saved))
        progressmessage.update()
        page.update()

    # ★追加：出力フォルダ内の全txtを1ファイルに結合する
    def merge_texts(e):
        outputpath = selected_output_path.value
        if not outputpath:
            progressmessage.value = "出力先フォルダが選択されていません"
            progressmessage.update()
            return
        txt_files = sorted(glob.glob(os.path.join(outputpath, "*.txt")))
        # _merged.txt 自体は除外
        txt_files = [f for f in txt_files if not os.path.basename(f).startswith("_merged")]
        if not txt_files:
            progressmessage.value = "結合するテキストファイルが見つかりません"
            progressmessage.update()
            return
        merged_path = os.path.join(outputpath, "_merged.txt")
        with open(merged_path, "w", encoding="utf-8") as outf:
            for txt_file in txt_files:
                basename = os.path.basename(txt_file)
                outf.write("【{}】\n".format(basename))
                with open(txt_file, "r", encoding="utf-8") as inf:
                    outf.write(inf.read())
                outf.write("\n\n")
        progressmessage.value = "結合完了: _merged.txt ({} ファイル)".format(len(txt_files))
        progressmessage.update()
        page.update()

    def handle_dlg_modal_close(e):
        config_obj={
            "json":chkbx_json.value,
            "txt":chkbx_txt.value,
            "xml":chkbx_xml.value,
            "tei":chkbx_tei.value,
            "pdf":chkbx_pdf.value,
            "pdf_viztxt":chkbx_pdf_viztxt.value,
        }
        with open('userconf.yaml','w')as wf:
            yaml.dump(config_obj, wf, default_flow_style=False, allow_unicode=True)
        page.close(dlg_modal)

    def change_pdfstatus(e):
        chkbx_pdf_viztxt.disabled=not chkbx_pdf.value
        chkbx_pdf_viztxt.update()

    preview_image=ft.Image(src="dummy.dat", fit=ft.ImageFit.CONTAIN, expand=True)

    # ★変更：ft.Text → ft.TextField（編集可能なテキストエリア）
    preview_text=ft.TextField(
        value="",
        multiline=True,
        min_lines=10,
        text_size=13,
        border_color=ft.colors.BLUE_200,
        hint_text="OCR結果がここに表示されます（直接編集できます）",
        expand=True,
    )

    pick_directory_dialog = ft.FilePicker(on_result=pick_directory_result)
    pick_output_dialog = ft.FilePicker(on_result=pick_output_result)
    pick_files_dialog = ft.FilePicker(on_result=pick_files_result)
    progressbar = ft.ProgressBar(width=400,value=0)
    selected_input_path = ft.Text()
    selected_output_path = ft.Text()
    progressmessage=ft.Text()
    chkbx_visualize = ft.Checkbox(label="認識箇所の可視化画像を保存する", value=True)
    chkbx_json = ft.Checkbox(label="JSON形式", value=True)
    chkbx_txt = ft.Checkbox(label="TXT形式", value=True)
    chkbx_xml = ft.Checkbox(label="XML形式", value=True)
    chkbx_tei = ft.Checkbox(label="TEI形式", value=False)
    chkbx_pdf = ft.Checkbox(label="透過テキスト付きPDF(ベータ)", value=False,on_change=change_pdfstatus)
    chkbx_pdf_viztxt = ft.Checkbox(label="PDFに可視化でテキストを乗せる", value=True)
    if os.path.exists("userconf.yaml"):
        with open('userconf.yaml', encoding='utf-8')as f:
            config_obj= yaml.safe_load(f)
            if "json" in config_obj:
                chkbx_json.value=config_obj["json"]
            if "xml" in config_obj:
                chkbx_xml.value=config_obj["xml"]
            if "tei" in config_obj:
                chkbx_tei.value=config_obj["tei"]
            if "txt" in config_obj:
                chkbx_txt.value=config_obj["txt"]
            if "pdf" in config_obj:
                chkbx_pdf.value=config_obj["pdf"]
            if "pdf_viztxt" in config_obj:
                chkbx_pdf_viztxt.value=config_obj["pdf_viztxt"]
                chkbx_pdf_viztxt.disabled=not chkbx_pdf.value

    page.overlay.extend([pick_files_dialog,pick_directory_dialog,pick_output_dialog])
    file_upload_btn=ft.ElevatedButton(
                    "画像ファイルを処理する",
                    icon=ft.icons.UPLOAD_FILE,
                    on_click=lambda _: pick_files_dialog.pick_files(
                        allow_multiple=False
                    ),
                )
    directory_upload_btn=ft.ElevatedButton(
                    "フォルダ内の画像を処理する",
                    icon=ft.icons.FOLDER_OPEN,
                    on_click=lambda _: pick_directory_dialog.get_directory_path(),
                )
    directory_output_btn=ft.ElevatedButton(
                    "出力先を選択する",
                    on_click=lambda _: pick_output_dialog.get_directory_path(),
                )
    ocr_btn=ft.ElevatedButton(text="OCR",
                                 on_click=ocr_button_result,
                                 style=ft.ButtonStyle(
                                    padding=30,
                                    shape=ft.RoundedRectangleBorder(radius=10)
                                    ),
                                 disabled=True)

    # ★追加：テキスト保存ボタン
    save_text_btn = ft.ElevatedButton(
        text="編集内容を保存",
        icon=ft.icons.SAVE,
        on_click=save_text,
        disabled=True,
        tooltip="表示中のテキストを編集してtxt・xml・jsonに保存できます",
    )

    # ★追加：全ページテキスト保存ボタン
    merge_text_btn = ft.ElevatedButton(
        text="全ページのテキストを保存",
        icon=ft.icons.MERGE_TYPE,
        on_click=merge_texts,
        disabled=True,
        tooltip="出力フォルダ内の全txtファイルを_merged.txtにまとめて保存します",
    )

    # 画像ページ選択プルダウン
    page_dropdown = ft.Dropdown(
        width=300,
        options=[],
        disabled=True,
        on_change=lambda e: jump_to_page(e),
    )

    def jump_to_page(e):
        nonlocal preview_index
        if page_dropdown.value is not None:
            preview_index = int(page_dropdown.value)
            if len(visualizepathlist) > 0:
                preview_image.src = visualizepathlist[preview_index]
            else:
                preview_image.src = inputpathlist[preview_index]
            preview_text.value = outputtxtlist[preview_index]
            page.update()

    def update_dropdown():
        page_dropdown.options = [
            ft.dropdown.Option(key=str(i), text=os.path.basename(inputpathlist[i]))
            for i in range(len(inputpathlist))
        ]
        if len(inputpathlist) > 0:
            page_dropdown.value = "0"
            page_dropdown.disabled = False
        page_dropdown.update()

    preview_image_int=ft.InteractiveViewer(
            min_scale=0.5,
            max_scale=10,
            boundary_margin=ft.margin.all(20),
            content=preview_image,
            expand=True,
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
    )

    preview_prev_btn=ft.ElevatedButton(text="< 前の画像", on_click=prev_image, disabled=True)
    preview_next_btn=ft.ElevatedButton(text="次の画像 >", on_click=next_image, disabled=True)
    customize_btn=ft.ElevatedButton("出力形式の選択", on_click=lambda e: page.open(dlg_modal))
    dlg_modal = ft.AlertDialog(
        modal=True,
        title=ft.Text("設定"),
        content=ft.Text("出力形式を選択してください"),
        actions=[
            chkbx_txt,
            chkbx_json,
            ft.Row([chkbx_xml,chkbx_tei]),
            ft.Row([chkbx_pdf,chkbx_pdf_viztxt]),
            ft.TextButton("OK", on_click=handle_dlg_modal_close),
        ],
        actions_alignment=ft.MainAxisAlignment.END,
    )
    page.add(
        ft.Row(
            [
                ft.Text("処理対象と出力先を選択して、「OCR」ボタンを押してください")
            ],
            ),
        ft.Divider(),
        ft.Row(
            [
                file_upload_btn,
                directory_upload_btn,
                ft.Text("処理対象："),
                selected_input_path,
            ]
        ),
        ft.Divider(),
        ft.Row(
            [
                directory_output_btn,
                ft.Text("出力先："),
                selected_output_path,
            ]
        ),
        ft.Divider(),
        ft.Row(
            [ocr_btn,
             ft.Column([chkbx_visualize,customize_btn
                        ]),
             ft.Column([progressmessage,progressbar]),
            ]
        ),
        ft.Divider(),
        # ★OCR結果タイトル（目立つように）
        ft.Text("OCR結果", size=20, weight=ft.FontWeight.BOLD),
        ft.Divider(),
        # ★ページナビゲーション行
        ft.Row([
            preview_prev_btn,
            ft.Text("画像を選択："),
            page_dropdown,
            preview_next_btn,
        ]),
        ft.Divider(),
        # ★画像とテキストを横並びに配置
        ft.Row(
            controls=[
                # 左：画像エリア
                ft.Container(
                    content=ft.InteractiveViewer(
                        min_scale=0.5,
                        max_scale=10,
                        boundary_margin=ft.margin.all(20),
                        content=preview_image,
                        expand=True,
                        clip_behavior=ft.ClipBehavior.HARD_EDGE,
                    ),
                    expand=3,
                    height=700,
                    border=ft.border.all(1, ft.colors.GREY_300),
                    clip_behavior=ft.ClipBehavior.HARD_EDGE,
                    padding=0,
                ),
                # 右：テキストエリア
                ft.Container(
                    content=ft.Column(
                        controls=[
                            preview_text,
                            ft.Row([save_text_btn, merge_text_btn]),
                        ],
                        scroll=ft.ScrollMode.ALWAYS,
                    ),
                    expand=1,
                    height=700,
                    border=ft.border.all(1, ft.colors.BLUE_200),
                    padding=8,
                ),
            ],
            expand=True,
            vertical_alignment=ft.CrossAxisAlignment.START,
        ),
    )
ft.app(main)
