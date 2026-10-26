# NVFlare PR #5554 UI screenshots

Actual Chrome captures of the Dashboard served by a temporary Flask backend from commit `af89b73cbdbdceeb19779618ccd3fa4639731550`. All accounts and project data are synthetic. Images were captured at a 1120 x 680 browser viewport with a device scale factor of 2.

- `01-registration-organization-error.png`: invalid organization rejected with HTTP 400, error shown inline, entered value retained and field editable.
- `02-registration-corrected.png`: correcting the organization to `Test_Holger_Client` succeeds with HTTP 200 and completes registration.
- `03-server-name-error.png`: invalid hostname rejected with HTTP 400 and a specific inline error.

This branch hosts review evidence for https://github.com/NVIDIA/NVFlare/pull/5554. It is separate from the implementation branch and is not intended to be merged. The images are unmodified browser captures; native browser alert dialogs are not pictured.
