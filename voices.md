# Voice preference by language

For each target language, voices are listed best first, separately for female (F) and male (M) speakers. The first
unused voice of the speaker's gender is assigned.

## Basis

- **Hindi, from listening:** Suhani and Amit (Hindi) are the best. After them: Gujarati (Dhara, Parth),
  Saba (Urdu), Manpreet (Punjabi), Vaidehi (Maithili), Dogri (Preeti, Sham) and Rano (Sindhi). Kavya (Hindi)
  is bad and is left out everywhere.
- **Assamese, Bengali and Odia sound alike:** their voices can stand in for each other.
- **Tamil, Telugu, Kannada and Malayalam (Dravidian) sound alike:** their voices can stand in for each other.
- **Languages not discussed (my choices):** a language's own voices go first. Next come voices from its close
  group, and then the Hindi list, which is the general fallback because it was the one checked by ear:
  - Marathi, Konkani, Nepali, Sanskrit, Kashmiri, Punjabi, Urdu, Sindhi, Gujarati, Maithili and Dogri are
    Indo-Aryan languages of the north and west, so the Hindi list comes straight after their own voices.
  - Manipuri, Bodo and Santali are spoken in the north-east and east, so the Assamese, Bengali and Odia voices
    come before the Hindi list.
  - English has no voices of its own, so it uses the Hindi list.

Hindi list — F: Suhani, Dhara, Saba, Vaidehi, Preeti · M: Amit, Parth, Manpreet, Sham, Rano

## Ranks

| Language | Female | Male |
|---|---|---|
| Hindi (hi) | Suhani, Dhara, Saba, Vaidehi, Preeti | Amit, Parth, Manpreet, Sham, Rano |
| English (en) | Suhani, Dhara, Saba, Vaidehi, Preeti | Amit, Parth, Manpreet, Sham, Rano |
| Gujarati (gu) | Suhani, Dhara, Saba, Vaidehi, Preeti | Amit, Parth, Manpreet, Sham, Rano |
| Urdu (ur) | Saba, Suhani, Dhara, Vaidehi, Preeti | Zaid, Amit, Parth, Manpreet, Sham |
| Punjabi (pa) | Kaur, Suhani, Dhara, Saba, Vaidehi | Manpreet, Amit, Parth, Sham, Rano |
| Maithili (mai) | Vaidehi, Suhani, Dhara, Saba, Preeti | Madhukar, Amit, Parth, Manpreet, Sham |
| Dogri (doi) | Preeti, Suhani, Dhara, Saba, Vaidehi | Sham, Amit, Parth, Manpreet, Rano |
| Sindhi (sd) | Moomal, Saba, Suhani, Dhara, Vaidehi | Rano, Amit, Parth, Manpreet, Sham |
| Kashmiri (ks) | Zoon, Saba, Suhani, Dhara, Vaidehi | Ishfaq, Manpreet, Amit, Parth, Rano |
| Marathi (mr) | Anagha, Suhani, Dhara, Saba, Vaidehi | Chinmay, Amit, Parth, Manpreet, Sham |
| Konkani (kok) | Anjali, Anagha, Suhani, Dhara, Saba | Sandeep, Chinmay, Amit, Parth, Manpreet |
| Nepali (ne) | Srijana, Vaidehi, Suhani, Dhara, Saba | Sagar, Madhukar, Amit, Parth, Manpreet |
| Sanskrit (sa) | Bharati, Suhani, Dhara, Saba, Vaidehi | Aryaman, Amit, Parth, Manpreet, Sham |
| Assamese (as) | Prastuti, Ishita, Itishree, Suhani, Dhara | Ankur, Sourav, Akash, Amit, Parth |
| Bengali (bn) | Ishita, Prastuti, Itishree, Suhani, Dhara | Sourav, Ankur, Akash, Amit, Parth |
| Odia (or) | Itishree, Ishita, Prastuti, Suhani, Dhara | Akash, Sourav, Ankur, Amit, Parth |
| Manipuri (mni) | Thoibi, Prastuti, Ishita, Itishree, Suhani | Chaoba, Ankur, Sourav, Akash, Amit |
| Bodo (brx) | Gwrbw, Prastuti, Ishita, Itishree, Suhani | Sansuma, Ankur, Sourav, Akash, Amit |
| Santali (sat) | Phulmani, Itishree, Ishita, Prastuti, Suhani | Sibu, Akash, Sourav, Ankur, Amit |
| Tamil (ta) | Anitha, Lakshmi, Sravani, Deepika, Suhani | Arun, Kiran, Vamsi, Adarsh, Amit |
| Malayalam (ml) | Lakshmi, Anitha, Deepika, Sravani, Suhani | Kiran, Arun, Adarsh, Vamsi, Amit |
| Telugu (te) | Sravani, Deepika, Anitha, Lakshmi, Suhani | Vamsi, Adarsh, Arun, Kiran, Amit |
| Kannada (kn) | Deepika, Sravani, Lakshmi, Anitha, Suhani | Adarsh, Vamsi, Kiran, Arun, Amit |

Within the Dravidian group, the closest languages come first: Tamil with Malayalam, and Telugu with Kannada.
The table shows each language's top five per gender (`tts.PREFERENCE`). After them, every other voice of that
gender follows in one shared order (`tts.REST_ORDER`): the Hindi list, the other north and west voices (Zaid, Kaur,
Madhukar, Moomal, Zoon, Ishfaq, Anagha, Chinmay, Anjali, Sandeep, Srijana, Sagar, Bharati, Aryaman), then the east
(Ishita, Sourav, Prastuti, Ankur, Itishree, Akash), the north-east (Thoibi, Chaoba, Gwrbw, Sansuma, Phulmani, Sibu)
and the Dravidian voices (Anitha, Arun, Lakshmi, Kiran, Sravani, Vamsi, Deepika, Adarsh), and Kavya last.
The studio offers the first 8 voices of each gender in this order for Hindi, Tamil, Telugu, Malayalam, Bengali and
Marathi, and the first 4 for every other language (`tts.preferred`). Speakers get voices down that list, and they're
reused in the same order once all are taken. Each offered voice has a sample clip of the language's sentence
(`backend/samples.py`).

## To check by ear

- **Manipuri, Bodo and Santali:** is the Assamese, Bengali and Odia fallback better than the Hindi list for these?
- **The Dravidian group:** none of these voices has been heard on a Hindi text. They come before the Hindi list
  for Dravidian languages only.
