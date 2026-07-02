# missions/ — ラジオ連載 (Lab Series) のシリーズマニフェスト

このディレクトリは、.alog アーカイブと **連載の実験文脈** を同じリポで管理するための
シリーズマニフェスト置き場。ARIGATO RADIO (coffee-roast-agent) の連載収録
(Lab Series S01 形式: 仮説 → 実験 → 完成 → 販売) が、ここを唯一の共有ソースとして読む。

## なぜここに置くか

- 実験の正体 (どの .alog が対照でどれが挑戦か、仮説、判定) は焙煎データと不可分
- coffee-roast-agent と実験管理側 (money-platform) の両方が読める共有リポはここだけ
  (money-platform は私的データを含むため参照させない)
- 焙煎の集計値・実験メモのみで PII なし

## 読み書きの役割分担

| 誰が | 何をする |
|---|---|
| 実験管理側のセッション | エピソードの追加・仮説とトークテーマの記入・判定 (verdict) の記入 |
| coffee-roast-agent のセッション | **読むだけ**。`SNN.json` → `backend/events.json` の RADIO_BROADCAST エントリ + `backend/themes/` のテーマファイルに変換し、`run-reels-session` で収録 |

## スキーマ (SNN.json)

```jsonc
{
  "series_id": "S02",
  "series_title_ja": "...",        // シーズンの題
  "premise_ja": "...",             // シーズンの前提 (1段落)
  "episodes": [
    {
      "id": "S02-E01",             // events.json の evt_radio_{id}_{date} に対応させる
      "status": "ready",           // planned | ready (alogあり) | recorded | published
      "alog": "ファイル名.alog",    // このリポのルート相対。planned なら null
      "reference_alogs": ["..."],  // 実況が参照してよい過去ロット (任意)
      "title_ja": "...",
      "hypothesis_ja": "...",      // このエピソードで検証する1変数の仮説
      "talk_theme": {
        "angle_ja": "...",         // 実況の角度 (1-2文)
        "beats_ja": ["..."]        // 触れてほしい論点 (3-5個)
      },
      "verdict": null,             // 判定後に記入 (例: "WIN: 余韻は新ロット")
      "next_hook_ja": "..."        // 次回への引き
    }
  ]
}
```

## 運用ルール

- 1エピソード = 1実験 = **1変数**。判定はブラインド二択1問
- `verdict` は飲み比べ後にのみ記入 (未実施は null のまま)
- エピソードの追加・編集はこのリポへの通常 commit で行う
