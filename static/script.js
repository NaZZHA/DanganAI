file_upload_button = document.getElementById('upload_icon_button')
load_json_file = document.getElementById('load_json')
json_input_field = document.getElementById('upload_savefile')


file_upload_button.addEventListener('click', () => {
	document.getElementById('upload_icon').click();
});

load_json_file.addEventListener('click', ()=> {
	document.getElementById('upload_savefile').click();
});

json_input_field.addEventListener('change', ()=> {
	if(json_input_field.value){
		console.log(json_input_field.value)
		document.getElementById('submit_json_file').click();
	}
})