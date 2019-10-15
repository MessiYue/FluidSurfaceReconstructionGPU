#version 150

uniform vec3	wLightPos;
uniform vec3	wCamPos;
uniform mat4 MV;
uniform vec3	ambient = vec3(0.20f, 0.20f, 0.20f);
uniform vec3	specular = vec3(0.8f, 0.8f, 0.8f);
uniform float	shininess = 1.0f;
uniform bool	lighting = true;

// SSAO
uniform sampler2D   ssaoTexture;
uniform bool        ssaoEnabled = false;

in  vec3 vsPosition;
in  vec3 vsColor;
in  vec3 vsNormal;

out vec4 outputF;

void main(void) {

	vec3 norm;
	norm.xy = gl_PointCoord.xy * vec2(2.0, -2.0) + vec2(-1.0,1.0);
	float mag = dot(norm.xy, norm.xy);
	if(mag > 1.0) discard;
	norm.z = sqrt(1.0 - mag);

    vec3 view_dir = normalize(wCamPos - vsPosition);	// compute view direction and normalize it
    vec3 normal = norm;
    vec3 color = vsColor;
    if (lighting) 
	{
		vec3 lightPos = (MV * vec4(wLightPos, 1.0)).xyz;
        vec3 light_dir = normalize(lightPos);
        float df = max(dot(light_dir, normal), 0);
        float sf = 0.0;	// specular factor
        if (df > 0.0) {	// if the vertex is lit compute the specular color
            vec3 half_vector = normalize(light_dir + view_dir);		// compute the half vector
            sf = max(dot(half_vector, normal), 0.0);
            sf = pow(sf, shininess);
        }

        color = color * df + specular * sf + ambient * color;
        if (ssaoEnabled) {
            vec2 texCoord = gl_FragCoord.xy / textureSize(ssaoTexture, 0);
            float coeff = texture(ssaoTexture, texCoord).r;
            color = color * coeff;
        }
		color.r = pow(color.r, 1/2.2f);
		color.g = pow(color.g, 1/2.2f);
		color.b = pow(color.b, 1/2.2f);
    }
    else
        color = color + ambient;

    outputF = vec4(color, 1.0f);
}
