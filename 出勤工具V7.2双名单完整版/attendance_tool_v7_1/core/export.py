import io
import pandas as pd
from openpyxl.utils import get_column_letter

def make_excel(df):

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        df.to_excel(
            writer,
            index=False,
            sheet_name="出勤结果"
        )

        worksheet = writer.book[
            "出勤结果"
        ]

        for index, column in enumerate(df.columns, 1):
            worksheet.column_dimensions[get_column_letter(index)].width = 26 if column in ("游戏 ID", "YY ID", "人员") else 12

    output.seek(0)

    return output
